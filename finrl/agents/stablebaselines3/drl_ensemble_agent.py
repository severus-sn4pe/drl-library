from __future__ import annotations

import os
import shutil
import time
from datetime import timedelta, datetime
from multiprocessing import Process

import numpy as np
import pandas as pd

from config import crypto
from finrl.agents.stablebaselines3.tensorboard_callback import TensorboardCallback
from finrl.meta.env_custom.env_custom import CustomTradingEnv
from finrl.meta.env_custom.random_init import RandomInit
from lib.drl import data_split, load_model_from_file
from lib.logger import log_duration_from_start


def log_df_info(name, df):
    print(f"{name:7s} start={df.iloc[0]['date']} end={df.iloc[-1]['date']} shape={df.shape} ")


class DRLEnsembleAgent:
    @staticmethod
    def get_train_env(df, env_kwargs) -> CustomTradingEnv:
        return DRLEnsembleAgent.get_env(df, env_kwargs, "train")

    @staticmethod
    def get_val_env(df, env_kwargs):
        return DRLEnsembleAgent.get_env(df, env_kwargs, "validation")

    @staticmethod
    def get_test_env(df, env_kwargs):
        return DRLEnsembleAgent.get_env(df, env_kwargs, "test")

    @staticmethod
    def get_env(df, env_kwargs, mode):
        kwargs = env_kwargs.copy()
        kwargs['mode'] = mode
        e_gym = CustomTradingEnv(df=df, **kwargs)
        env, _ = e_gym.get_sb_env()
        return env

    @staticmethod
    def train_model(model, total_timesteps, tb_log_name, target_model_filename, model_name):
        start = time.time()
        trained_model = model.learn(total_timesteps=total_timesteps,
                                    tb_log_name=tb_log_name,
                                    callback=[TensorboardCallback()])

        # print(f"Storing model in {target_model_filename}")
        trained_model.save(target_model_filename)
        log_duration_from_start(start, f"Training {model_name}")
        # log_finished(True, start, None, None, None, send_discord=False)
        return trained_model

    @staticmethod
    def get_validation_sharpe(state_filename):
        """Calculate Sharpe ratio based on validation results"""
        df_state = pd.read_csv(state_filename)
        df_state['daily_return'] = df_state["account_value"].pct_change(1)
        # If the agent did not make any transaction
        if df_state["daily_return"].var() == 0:
            if df_state["daily_return"].mean() > 0:
                return np.inf
            else:
                return 0.0
        else:
            return (
                    (365 ** 0.5)
                    * df_state["daily_return"].mean()
                    / df_state["daily_return"].std()
            )

    def __init__(self, df, stock_dim, initial_amount, initial_num_stock_shares,
                 buy_cost_pct, sell_cost_pct, reward_scaling,
                 tech_indicator_list, print_verbosity, make_plots,
                 strategy_name, run_name, model_settings,
                 start_dates, windows, iterations,
                 root_dir, trained_model_dir, results_dir, tensorboard_log_dir, res):

        self.df = df

        self.train_start = start_dates['train']
        self.val_start = start_dates['val']
        self.test_start = start_dates['test']
        self.train_window = windows['train']
        self.val_window = windows['val']
        self.iterations = iterations

        self.tech_indicator_list = tech_indicator_list
        self.stock_dim = stock_dim
        self.initial_amount = initial_amount
        self.initial_num_stock_shares = initial_num_stock_shares
        self.buy_cost_pct = [buy_cost_pct] * self.stock_dim
        self.sell_cost_pct = [sell_cost_pct] * self.stock_dim
        self.reward_scaling = reward_scaling
        self.state_space = 1 + 2 * self.stock_dim + len(self.tech_indicator_list) * self.stock_dim
        self.action_space = self.stock_dim
        print(f"Stock Dimension: {self.stock_dim}, State Space: {self.state_space}")

        self.print_verbosity = print_verbosity
        self.make_plots = make_plots

        self.skip_train_first = True
        self.skip_train_all = False  # Set true if absolutely no training should be done, init model is used for all
        self.skip_retrain = False  # set true if no retraining should be done with the winner model per iteration

        self.strategy_name = strategy_name
        self.run_name = run_name
        self.model_settings = model_settings

        self.root_dir = root_dir
        self.trained_model_dir = f"{root_dir}/{trained_model_dir}"
        self.results_dir = f"{root_dir}/{results_dir}"
        self.tb_log_dir = f"{root_dir}/{tensorboard_log_dir}"
        self.results_base = f"{self.results_dir}/{self.strategy_name}/{self.run_name}"
        self.run_model_dir = f"{trained_model_dir}/{self.strategy_name}/{self.run_name}"

        self.iteration_limits = self.setup_iterations()
        self.res = res

    def __check_directory_structure(self):
        if not os.path.exists(self.run_model_dir):
            os.mkdir(self.run_model_dir)

        if not os.path.exists(self.results_base):
            os.mkdir(self.results_base)

    def setup_iterations(self):
        iterations = []
        start_train = datetime.strptime(self.train_start, "%d.%m.%Y")
        start_val = datetime.strptime(self.val_start, "%d.%m.%Y")
        start_test = datetime.strptime(self.test_start, "%d.%m.%Y")

        validation_delta = timedelta(days=self.val_window)
        validation_delta_end = timedelta(days=self.val_window - 1, hours=23, minutes=59, seconds=59)
        train_delta_end = timedelta(days=self.train_window - 1, hours=23, minutes=59, seconds=59)

        for i in range(self.iterations):
            end_val = start_val + validation_delta_end
            end_test = start_test + validation_delta_end
            end_train = start_train + train_delta_end
            start_retrain = start_train + validation_delta
            end_retrain = end_val
            iterations.append(
                {'train_start': start_train, 'train_end': end_train,
                 'val_start': start_val, 'val_end': end_val,
                 'test_start': start_test, 'test_end': end_test,
                 'retrain_start': start_retrain, 'retrain_end': end_retrain})
            start_val += validation_delta
            start_test += validation_delta
            start_train += validation_delta
        iterations = pd.DataFrame(iterations)
        iterations.iloc[crypto.VALIDATION_ITERATIONS - 2]['test_end'] = '2022-12-01 23:59:59'  # TODO: improve
        return iterations

    def get_dataset_windows(self, res, iteration: int = 0):
        if iteration >= self.iterations:
            raise ValueError(f"max allowed iteration is {self.iterations - 1}, but {iteration} was provided")

        start_format = "%Y-%m-%d" if res == '1d' else "%Y-%m-%d %X"
        end_format = "%Y-%m-%d %X"
        window = {
            "train_start": self.iteration_limits.iloc[iteration]['train_start'].strftime(start_format),
            "train_end": self.iteration_limits.iloc[iteration]['train_end'].strftime(end_format),
            "val_start": self.iteration_limits.iloc[iteration]['val_start'].strftime(start_format),
            "val_end": self.iteration_limits.iloc[iteration]['val_end'].strftime(end_format),
            "test_start": self.iteration_limits.iloc[iteration]['test_start'].strftime(start_format),
            "test_end": self.iteration_limits.iloc[iteration]['test_end'].strftime(end_format),
            "retrain_start": self.iteration_limits.iloc[iteration]['retrain_start'].strftime(start_format),
            "retrain_end": self.iteration_limits.iloc[iteration]['retrain_end'].strftime(end_format),
        }
        return window

    def get_datasets(self, iteration, res):
        dataset_windows = self.get_dataset_windows(res, iteration)
        train_df = data_split(self.df, dataset_windows['train_start'], dataset_windows['train_end'])
        val_df = data_split(self.df, dataset_windows['val_start'], dataset_windows['val_end'])
        test_df = data_split(self.df, dataset_windows['test_start'], dataset_windows['test_end'])
        retrain_df = data_split(self.df, dataset_windows['retrain_start'], dataset_windows['retrain_end'])

        print(f"Loaded {res} datasets for iteration {iteration}:")
        log_df_info("train", train_df)
        log_df_info("val", val_df)
        log_df_info("test", test_df)
        log_df_info("retrain", retrain_df)

        return train_df, val_df, test_df, retrain_df

    def prediction_validation(self, model, environment, deterministic=True):
        # try:
        #     test_env, test_obs = environment.get_sb_env()
        # except ValueError:
        #     test_env, test_obs = environment.get_sb_env()
        actions_memory = []
        test_obs = environment.reset()

        unique_trading_days = len(environment.envs[0].df.index.unique())
        for i in range(unique_trading_days):
            action, _states = model.predict(test_obs, deterministic=deterministic)
            test_obs, rewards, dones, info = environment.step(action)
            if i == (unique_trading_days - 2):
                state_memory = environment.env_method(method_name="save_state_memory")
                actions_memory = environment.env_method(method_name="save_action_memory")
            if dones[0]:
                # print(f"validation done in step {i}!")
                break
        return state_memory[0], actions_memory[0]

    def get_env_kwargs(self, model_name, mode):
        env_kwargs = {
            "hmax": 10_000,
            "initial_amount": self.initial_amount,
            "num_stock_shares": self.initial_num_stock_shares,
            "buy_cost_pct": self.buy_cost_pct,
            "sell_cost_pct": self.sell_cost_pct,
            "state_space": self.state_space,
            "stock_dim": self.stock_dim,
            "tech_indicator_list": self.tech_indicator_list,
            "action_space": self.action_space,
            "reward_scaling": self.reward_scaling,
            "make_plots": self.make_plots,
            "print_verbosity": self.print_verbosity,
            "mode": mode,
            "strategy_name": self.strategy_name,
            "run_name": self.run_name,
            "model_name": model_name,
            "random_init": RandomInit(random_init=False)
        }
        return env_kwargs

    def run_validation(self, test_name, env, model, name):
        start = time.time()
        df_state, df_actions = self.prediction_validation(model, env)
        log_duration_from_start(start, name)

        df_state.to_csv(f"{test_name}_state.csv")
        df_actions.to_csv(f"{test_name}_actions.csv")

    def load_last_state(self, iteration_results_prefix, ticker_names):
        state_file = pd.read_csv(f"{iteration_results_prefix}_state.csv", index_col=0)
        last_state = state_file.iloc[-1]
        num_shares = last_state[[x + "_amount" for x in ticker_names]].tolist()
        cash = last_state['cash']
        # print(f"cash          : {cash}")
        # print(f"NumStockShares: {num_shares}")
        return cash, num_shares

    def train_model_on_iteration(self, model_name, iteration, df):
        train_settings = self.get_env_kwargs(model_name, "train")
        train_settings['model_name'] = model_name

        print(f"Starting training {model_name} on iteration {iteration}")
        model_iteration_name = f"{self.run_name}_iter{iteration}_{model_name}"

        load_model_filename = f"{self.trained_model_dir}/{self.model_settings[model_name]['init_model']}"
        if iteration > 0:
            load_model_filename = f"{self.run_model_dir}/{self.run_name}_iter{iteration - 1}_{model_name}"
        save_model_filename = f"{self.run_model_dir}/{self.run_name}_iter{iteration}_{model_name}"

        train_env = self.get_train_env(df, train_settings)
        model_kwargs = self.model_settings[model_name]['params']

        loaded_model = load_model_from_file(model_name, load_model_filename, None,
                                            device=model_kwargs['device'])
        loaded_model.set_env(train_env)

        if iteration == 0 and self.skip_train_first:
            # print("Skip training at iteration 0 and copy model")
            shutil.copyfile(f"{load_model_filename}.zip", f"{save_model_filename}.zip")
        else:
            self.train_model(loaded_model, total_timesteps=self.model_settings[model_name]['timesteps'],
                             tb_log_name=model_iteration_name, target_model_filename=save_model_filename,
                             model_name=model_name)

    def retrain_best_model(self, iteration, best_model_name, df):
        load_retrain_name = f"{self.run_model_dir}/{self.run_name}_iter{iteration}_{best_model_name}"
        save_retrain_name = f"{self.run_model_dir}/{self.run_name}_iter{iteration}_winner"

        retrain_settings = self.get_env_kwargs(best_model_name, "train")
        retrain_settings['model_name'] = best_model_name
        retrain_env = self.get_train_env(df, retrain_settings)
        retrain_args = self.model_settings[best_model_name]['params']

        winner_model = load_model_from_file(best_model_name, load_retrain_name, None, device=retrain_args['device'])
        winner_model.set_env(retrain_env)
        self.train_model(winner_model, total_timesteps=self.model_settings[best_model_name]['timesteps'],
                         tb_log_name=None, target_model_filename=save_retrain_name,
                         model_name=f"Best-{best_model_name}")

    def validate_model_on_iteration(self, model_name, iteration, df, ticker_names):
        # print(f"Iteration={iteration} Validation Model={model_name}")

        model_iteration_name = f"{self.run_name}_iter{iteration}_{model_name}"
        save_model_filename = f"{self.run_model_dir}/{self.run_name}_iter{iteration}_{model_name}"
        model_kwargs = self.model_settings[model_name]['params']

        if iteration >= 2:
            trade_res_prefix = f"{self.results_base}/{self.run_name}_iter{iteration - 2}_ensemble"
            cash_val, num_shares_val = self.load_last_state(trade_res_prefix, ticker_names)
        else:
            cash_val, num_shares_val = self.initial_amount, self.initial_num_stock_shares

        validation_model = load_model_from_file(model_name, save_model_filename,
                                                tensorboard_log=None, device=model_kwargs['device'])
        val_env_kwargs = self.get_env_kwargs(model_name, "validation")
        val_env_kwargs['initial_amount'] = cash_val
        val_env_kwargs['num_stock_shares'] = num_shares_val
        val_env = self.get_val_env(df, val_env_kwargs)

        validation_prefix = f"{self.results_base}/{model_iteration_name}_val"

        self.run_validation(validation_prefix, val_env, validation_model, "validation")

    def test_winner_on_iteration(self, iteration, best_model_name, df, ticker_names):
        if iteration == 0:
            cash, num_shares = self.initial_amount, self.initial_num_stock_shares
        else:
            previous_prefix = f"{self.results_base}/{self.run_name}_iter{iteration - 1}_ensemble"
            cash, num_shares = self.load_last_state(previous_prefix, ticker_names)

        saved_retrain_name = f"{self.run_model_dir}/{self.run_name}_iter{iteration}_winner"
        ensemble_model = load_model_from_file(best_model_name, saved_retrain_name,
                                              tensorboard_log=None,
                                              device=self.model_settings[best_model_name]['params']['device'])

        test_env_kwargs = self.get_env_kwargs(best_model_name, "test")

        test_prefix = f"{self.results_base}/{self.run_name}_iter{iteration}_ensemble"

        test_env_kwargs['initial_amount'] = cash
        test_env_kwargs['num_stock_shares'] = num_shares
        test_env = self.get_test_env(df, test_env_kwargs)
        self.run_validation(test_prefix, test_env, ensemble_model, "test")

    def run_ensemble(self):
        sharpe_list = {model: [] for model in self.model_settings["models"]}
        sharpe_list['test'] = []
        model_use = []

        temp_df, _, _, _ = self.get_datasets(0, self.res)
        ticker_names = temp_df.tic.unique().tolist()

        self.__check_directory_structure()
        main_start = time.time()

        for iteration in range(self.iterations):
            iteration_start = time.time()
            iteration_validation_sharpes = []
            print(f"\nIteration {iteration} / {self.iterations - 1}")
            train_df, val_df, test_df, retrain_df = self.get_datasets(iteration, self.res)
            print("---------------------------------------------")

            # TRAINING
            train_processes = []
            for model_name in self.model_settings['models']:
                pt = Process(target=self.train_model_on_iteration, args=(model_name, iteration, train_df))
                pt.start()
                train_processes.append(pt)
                # self.train_model_on_iteration(model_name, iteration, train_df)

            for p in train_processes:
                p.join()

            # VALIDATION
            for model_name in self.model_settings['models']:
                self.validate_model_on_iteration(model_name, iteration, val_df, ticker_names)

                validation_prefix = f"{self.results_base}/{self.run_name}_iter{iteration}_{model_name}_val"
                current_sharpe = self.get_validation_sharpe(f"{validation_prefix}_state.csv")
                sharpe_list[model_name].append(current_sharpe)
                iteration_validation_sharpes.append(current_sharpe)
                print(f"Iteration {iteration} {model_name} sharpe={current_sharpe}")

            # Determine the best model by highest sharpe
            best_model_index = np.argmax(iteration_validation_sharpes)
            best_model_name = self.model_settings['models'][best_model_index]
            model_use.append(best_model_name)
            best_model_sharpe = sharpe_list[best_model_name][iteration]
            print(f"BestModel={best_model_name} Sharpe={best_model_sharpe}")

            # Retrain winner model
            self.retrain_best_model(iteration, best_model_name, retrain_df)

            # Testing retrained Model on Test-Dataset
            self.test_winner_on_iteration(iteration, best_model_name, test_df, ticker_names)
            test_prefix = f"{self.results_base}/{self.run_name}_iter{iteration}_ensemble"
            sharpe_list['test'].append(self.get_validation_sharpe(f"{test_prefix}_state.csv"))

            log_duration_from_start(iteration_start, f"Iteration {iteration}")
        log_duration_from_start(main_start, "Complete Ensemble Strategy")
        self.save_results(sharpe_list, model_use)

    def save_results(self, sharpe_list, models_used):
        # save results and concat iteration results into one file
        self.concat_actions_files(models_used)
        self.concat_state_files(models_used)

        ensemble_stats = sharpe_list.copy()
        ensemble_stats['winner'] = models_used
        pd.DataFrame(ensemble_stats).to_csv(f"{self.results_base}/all_stats.csv")

    def concat_actions_files(self, winner_models):
        # concat separate actions files from each iteration into one file for the whole run
        actions = pd.DataFrame()
        for iteration in range(self.iterations):

            filename = f"{self.results_base}/{self.run_name}_iter{iteration}_ensemble_actions.csv"
            df_actions_iter = pd.read_csv(f"{filename}")

            # add last day for each iteration
            last_date = (pd.to_datetime(df_actions_iter.iloc[-1]['date']) + timedelta(days=1)).strftime('%Y-%m-%d')
            last_row = pd.DataFrame([[last_date] + [0] * self.stock_dim + [-1] * self.stock_dim])
            last_row.columns = df_actions_iter.columns
            df_actions_iter = pd.concat([df_actions_iter, last_row])
            df_actions_iter['iteration'] = iteration
            df_actions_iter['model'] = winner_models[iteration]
            if actions.empty:
                actions = df_actions_iter
            else:
                actions = pd.concat([actions, df_actions_iter])
            # print(results_base)

        actions = actions.reset_index(drop=True)
        target_filename = f"{self.results_base}/all_actions.csv"
        actions.to_csv(target_filename, index=False)

    def concat_state_files(self, winner_models):
        # concat separate state files from each iteration into one file for the whole run
        states = pd.DataFrame()
        for iteration in range(self.iterations):
            filename = f"{self.results_base}/{self.run_name}_iter{iteration}_ensemble_state.csv"
            df_state_iter = pd.read_csv(filename, index_col=0)

            df_state_iter['date'] = df_state_iter.index
            df_state_iter = df_state_iter.reset_index(drop=True)
            df_state_iter.loc[0, 'date'] = (pd.to_datetime(df_state_iter.loc[1]['date']) + timedelta(days=-1)).strftime(
                '%Y-%m-%d')
            df_state_iter = df_state_iter.set_index('date', drop=True)
            df_state_iter['iteration'] = iteration
            df_state_iter['model'] = winner_models[iteration]

            if states.empty:
                states = df_state_iter
            else:
                states = pd.concat([states, df_state_iter])

        states.to_csv(f"{self.results_base}/all_state.csv")
