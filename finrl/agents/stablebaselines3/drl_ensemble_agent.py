from __future__ import annotations

import time

import numpy as np
import pandas as pd
from stable_baselines3.common.vec_env import DummyVecEnv

from finrl.agents.stablebaselines3.models import MODELS, MODEL_KWARGS, NOISE
from finrl.agents.stablebaselines3.tensorboard_callback import TensorboardCallback
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from lib.drl import data_split


class DRLEnsembleAgent:
    def get_model(self, model_name, env, policy="MlpPolicy", policy_kwargs=None, model_kwargs=None, seed=None,
                  verbose=1):

        if model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

        if model_kwargs is None:
            temp_model_kwargs = MODEL_KWARGS[model_name]
        else:
            temp_model_kwargs = model_kwargs.copy()

        if "action_noise" in temp_model_kwargs:
            n_actions = env.action_space.shape[-1]
            temp_model_kwargs["action_noise"] = NOISE[
                temp_model_kwargs["action_noise"]
            ](mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
        print(temp_model_kwargs)
        return MODELS[model_name](
            policy=policy,
            env=env,
            tensorboard_log=f"{self.tb_log_dir}/{model_name}",
            verbose=verbose,
            policy_kwargs=policy_kwargs,
            seed=seed,
            **temp_model_kwargs,
        )

    def train_model(self, model, model_name, tb_log_name, iter_num, total_timesteps=5000):
        model = model.learn(total_timesteps=total_timesteps, tb_log_name=tb_log_name, callback=TensorboardCallback())
        model.save(f"{self.trained_model_dir}/{model_name.upper()}_{total_timesteps // 1000}k_{iter_num}")
        return model

    def get_validation_sharpe(self, iteration, model_name):
        """Calculate Sharpe ratio based on validation results"""
        df_total_value = pd.read_csv(f"{self.results_dir}/account_value_validation_{model_name}_{iteration}.csv")
        # If the agent did not make any transaction
        if df_total_value["daily_return"].var() == 0:
            if df_total_value["daily_return"].mean() > 0:
                return np.inf
            else:
                return 0.0
        else:
            return (
                    (4 ** 0.5)
                    * df_total_value["daily_return"].mean()
                    / df_total_value["daily_return"].std()
            )

    def __init__(self, df, train_period, val_test_period, rebalance_window, validation_window, stock_dim,
                 hmax, initial_amount, buy_cost_pct, sell_cost_pct, reward_scaling, state_space,
                 action_space, tech_indicator_list, print_verbosity,
                 root_dir, trained_model_dir, results_dir, tensorboard_log_dir):

        self.df = df
        self.train_period = train_period
        self.val_test_period = val_test_period

        self.unique_trade_date = df[
            (df.date > val_test_period[0]) & (df.date <= val_test_period[1])
            ].date.unique()
        self.rebalance_window = rebalance_window
        self.validation_window = validation_window

        self.stock_dim = stock_dim
        self.hmax = hmax
        self.initial_amount = initial_amount
        self.buy_cost_pct = buy_cost_pct
        self.sell_cost_pct = sell_cost_pct
        self.reward_scaling = reward_scaling
        self.state_space = state_space
        self.action_space = action_space
        self.tech_indicator_list = tech_indicator_list
        self.print_verbosity = print_verbosity
        self.results_dir = f"{root_dir}/{results_dir}"
        self.trained_model_dir = f"{root_dir}/{trained_model_dir}"
        self.root_dir = root_dir
        self.tb_log_dir = f"{root_dir}/{tensorboard_log_dir}"

    def DRL_validation(self, model, test_data, test_env, test_obs):
        """validation process"""
        for _ in range(len(test_data.index.unique())):
            action, _states = model.predict(test_obs)
            test_obs, rewards, dones, info = test_env.step(action)

    def DRL_prediction(self, model, name, last_state, iter_num, turbulence_threshold, initial):
        """make a prediction based on trained model"""

        start_date = self.unique_trade_date[iter_num - self.rebalance_window]
        end_date = self.unique_trade_date[iter_num]

        trade_data = data_split(self.df, start=start_date, end=end_date)
        trade_env = DummyVecEnv(
            [
                lambda: StockTradingEnv(
                    df=trade_data,
                    stock_dim=self.stock_dim,
                    hmax=self.hmax,
                    initial_amount=self.initial_amount,
                    num_stock_shares=[0] * self.stock_dim,
                    buy_cost_pct=[self.buy_cost_pct] * self.stock_dim,
                    sell_cost_pct=[self.sell_cost_pct] * self.stock_dim,
                    reward_scaling=self.reward_scaling,
                    state_space=self.state_space,
                    action_space=self.action_space,
                    tech_indicator_list=self.tech_indicator_list,
                    turbulence_threshold=turbulence_threshold,
                    initial=initial,
                    previous_state=last_state,
                    model_name=name,
                    mode="trade",
                    iteration=iter_num,
                    print_verbosity=self.print_verbosity,
                    root_dir=self.root_dir
                )
            ]
        )

        trade_obs = trade_env.reset()

        for i in range(len(trade_data.index.unique())):
            action, _states = model.predict(trade_obs)
            trade_obs, rewards, dones, info = trade_env.step(action)
            if i == (len(trade_data.index.unique()) - 2):
                # print(env_test.render())
                last_state = trade_env.render()

        df_last_state = pd.DataFrame({"last_state": last_state})
        df_last_state.to_csv(f"{self.results_dir}/last_state_{name}_{i}.csv", index=False)
        return last_state

    def get_train_env(self, end_date):
        train = data_split(self.df, start=self.train_period[0], end=end_date)
        train_env = DummyVecEnv([
            lambda: StockTradingEnv(
                df=train,
                stock_dim=self.stock_dim,
                hmax=self.hmax,
                initial_amount=self.initial_amount,
                num_stock_shares=[0] * self.stock_dim,
                buy_cost_pct=[self.buy_cost_pct] * self.stock_dim,
                sell_cost_pct=[self.sell_cost_pct] * self.stock_dim,
                reward_scaling=self.reward_scaling,
                state_space=self.state_space,
                action_space=self.action_space,
                tech_indicator_list=self.tech_indicator_list,
                print_verbosity=self.print_verbosity,
                root_dir=self.root_dir
            )
        ])
        return train_env

    def get_val_env(self, val_df, model_name, turbulence_threshold, iteration):
        val_env = DummyVecEnv([
            lambda: StockTradingEnv(
                df=val_df,
                stock_dim=self.stock_dim,
                hmax=self.hmax,
                initial_amount=self.initial_amount,
                num_stock_shares=[0] * self.stock_dim,
                buy_cost_pct=[self.buy_cost_pct] * self.stock_dim,
                sell_cost_pct=[self.sell_cost_pct] * self.stock_dim,
                reward_scaling=self.reward_scaling,
                state_space=self.state_space,
                action_space=self.action_space,
                tech_indicator_list=self.tech_indicator_list,
                turbulence_threshold=turbulence_threshold,
                iteration=iteration,
                model_name=model_name,
                mode="validation",
                print_verbosity=self.print_verbosity,
                root_dir=self.root_dir
            )
        ])
        return val_env

    def run_ensemble_strategy(self, A2C_model_kwargs, PPO_model_kwargs, DDPG_model_kwargs, timesteps_dict,
                              retrain=False):
        """Ensemble Strategy that combines PPO, A2C and DDPG"""
        print("============Start Ensemble Strategy============")
        # for ensemble model, it's necessary to feed the last state
        # of the previous model to the current model as the initial state
        last_state_ensemble = []

        ppo_sharpe_list = []
        ddpg_sharpe_list = []
        a2c_sharpe_list = []

        model_use = []
        validation_start_date_list = []
        validation_end_date_list = []
        iteration_list = []

        insample_turbulence = self.df[(self.df.date < self.train_period[1]) & (self.df.date >= self.train_period[0])]
        insample_turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 0.90)

        start = time.time()
        for i in range(self.rebalance_window + self.validation_window, len(self.unique_trade_date),
                       self.rebalance_window):

            val_start_index = i - self.rebalance_window - self.validation_window
            validation_start_date = self.unique_trade_date[val_start_index]
            val_end_index = i - self.rebalance_window
            validation_end_date = self.unique_trade_date[val_end_index]

            validation_start_date_list.append(validation_start_date)
            validation_end_date_list.append(validation_end_date)
            iteration_list.append(i)

            print("============================================")
            # initial state is empty
            if val_start_index == 0:
                # inital state
                initial = True
            else:
                # previous state
                initial = False
            print(f"Initial: {initial}")

            # TODO: recalculate turbulence index
            # Tuning turbulence index based on historical data
            # Turbulence lookback window is one quarter (63 days)
            end_date_index = self.df.index[self.df["date"] == self.unique_trade_date[val_start_index]].to_list()[-1]
            # start_date_index = end_date_index - 63 + 1
            #
            # historical_turbulence = self.df.iloc[start_date_index: (end_date_index + 1), :]
            # historical_turbulence = historical_turbulence.drop_duplicates(subset=["date"])
            # historical_turbulence_mean = np.mean(historical_turbulence.turbulence.values)
            # # print(historical_turbulence_mean)
            # if historical_turbulence_mean > insample_turbulence_threshold:
            #     # if the mean of the historical data is greater than the 90% quantile of insample turbulence data
            #     # then we assume that the current market is volatile,
            #     # therefore we set the 90% quantile of insample turbulence data as the turbulence threshold
            #     # meaning the current turbulence can't exceed the 90% quantile of insample turbulence data
            #     turbulence_threshold = insample_turbulence_threshold
            # else:
            #     # if the mean of the historical data is less than the 90% quantile of insample turbulence data
            #     # then we tune up the turbulence_threshold, meaning we lower the risk
            #     turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 1)

            turbulence_threshold = np.quantile(insample_turbulence.turbulence.values, 0.99)
            print("turbulence_threshold: ", turbulence_threshold)

            # ############# Environment Setup starts ##############
            train_env = self.get_train_env(self.unique_trade_date[val_start_index])
            validation = data_split(self.df, start=validation_start_date, end=validation_end_date)
            # ############# Environment Setup ends ##############

            # ############# Training and Validation starts ##############
            print(f"======Model training from: {self.train_period[0]} to {validation_start_date}")

            # print("==============Model Training===========")
            print("======A2C Training========")
            model_a2c = self.get_model("A2C", train_env, policy="MlpPolicy", model_kwargs=A2C_model_kwargs)
            model_a2c = self.train_model(model_a2c, "A2C", tb_log_name=f"A2C_{i}", iter_num=i,
                                         total_timesteps=timesteps_dict["A2C"])

            print(f"======A2C Validation from: {validation_start_date} to {validation_end_date}")
            val_env_a2c = self.get_val_env(validation, "A2C", turbulence_threshold, i)
            val_obs_a2c = val_env_a2c.reset()
            self.DRL_validation(model=model_a2c, test_data=validation, test_env=val_env_a2c, test_obs=val_obs_a2c)
            sharpe_a2c = self.get_validation_sharpe(i, model_name="A2C")
            print("A2C Sharpe Ratio: ", sharpe_a2c)

            print("======PPO Training========")
            model_ppo = self.get_model("PPO", train_env, policy="MlpPolicy", model_kwargs=PPO_model_kwargs)
            model_ppo = self.train_model(model_ppo, "PPO", tb_log_name=f"PPO_{i}", iter_num=i,
                                         total_timesteps=timesteps_dict["PPO"])

            print(f"======PPO Validation from: {validation_start_date} to {validation_end_date}")
            val_env_ppo = self.get_val_env(validation, "PPO", turbulence_threshold, i)
            val_obs_ppo = val_env_ppo.reset()
            self.DRL_validation(model=model_ppo, test_data=validation, test_env=val_env_ppo, test_obs=val_obs_ppo)
            sharpe_ppo = self.get_validation_sharpe(i, model_name="PPO")
            print("PPO Sharpe Ratio: ", sharpe_ppo)

            print("======DDPG Training========")
            model_ddpg = self.get_model("DDPG", train_env, policy="MlpPolicy", model_kwargs=DDPG_model_kwargs)
            model_ddpg = self.train_model(model_ddpg, "DDPG", tb_log_name=f"DDPG_{i}", iter_num=i,
                                          total_timesteps=timesteps_dict["DDPG"])

            print(f"======DDPG Validation from: {validation_start_date} to {validation_end_date}")
            val_env_ddpg = self.get_val_env(validation, "DDPG", turbulence_threshold, i)
            val_obs_ddpg = val_env_ddpg.reset()
            self.DRL_validation(model=model_ddpg, test_data=validation, test_env=val_env_ddpg, test_obs=val_obs_ddpg)
            sharpe_ddpg = self.get_validation_sharpe(i, model_name="DDPG")
            print("DDPG Sharpe Ratio: ", sharpe_ddpg)

            ppo_sharpe_list.append(sharpe_ppo)
            a2c_sharpe_list.append(sharpe_a2c)
            ddpg_sharpe_list.append(sharpe_ddpg)

            print(f"======Best Model Retraining from: {self.train_period[0]} to {validation_end_date}")
            # Environment setup for model retraining up to first trade date
            train_full_env = self.get_train_env(validation_end_date)
            # Model Selection based on sharpe ratio
            if (sharpe_ppo >= sharpe_a2c) & (sharpe_ppo >= sharpe_ddpg):
                used_model_name = "PPO"
                if retrain:
                    model_ensemble = self.get_model("PPO", train_full_env, policy="MlpPolicy",
                                                    model_kwargs=PPO_model_kwargs)
                    model_ensemble = self.train_model(model_ensemble, "ensemble", tb_log_name=f"ensemble_{i}",
                                                      iter_num=i, total_timesteps=timesteps_dict['PPO'])
                else:
                    model_ensemble = model_ppo
            elif (sharpe_a2c > sharpe_ppo) & (sharpe_a2c > sharpe_ddpg):
                used_model_name = "A2C"
                if retrain:
                    model_ensemble = self.get_model("A2C", train_full_env, policy="MlpPolicy",
                                                    model_kwargs=A2C_model_kwargs)
                    model_ensemble = self.train_model(model_ensemble, "ensemble", tb_log_name=f"ensemble_{i}",
                                                      iter_num=i, total_timesteps=timesteps_dict['A2C'])
                else:
                    model_ensemble = model_a2c
            else:
                used_model_name = "DDPG"
                if retrain:
                    model_ensemble = self.get_model("DDPG", train_full_env, policy="MlpPolicy",
                                                    model_kwargs=DDPG_model_kwargs)
                    model_ensemble = self.train_model(model_ensemble, "ensemble", tb_log_name="ensemble_{}".format(i),
                                                      iter_num=i, total_timesteps=timesteps_dict['DDPG'])
                else:
                    model_ensemble = model_ddpg
            model_use.append(used_model_name)
            # ############# Training and Validation ends ##############

            # ############# Trading starts ##############
            print(f"======Trading from: {validation_end_date} to {self.unique_trade_date[i]} with {used_model_name}")
            # print("Used Model: ", model_ensemble)
            last_state_ensemble = self.DRL_prediction(model=model_ensemble, name="ensemble",
                                                      last_state=last_state_ensemble, iter_num=i,
                                                      turbulence_threshold=turbulence_threshold, initial=initial)
            # ############# Trading ends ##############

        end = time.time()
        print("Ensemble Strategy took: ", (end - start) / 60, " minutes")

        df_summary = pd.DataFrame(
            [
                iteration_list,
                validation_start_date_list,
                validation_end_date_list,
                model_use,
                a2c_sharpe_list,
                ppo_sharpe_list,
                ddpg_sharpe_list,
            ]
        ).T
        df_summary.columns = ["Iter", "Val Start", "Val End", "Model Used", "A2C Sharpe", "PPO Sharpe", "DDPG Sharpe"]

        return df_summary
