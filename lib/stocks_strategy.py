import os

from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3

from finrl.agents.stablebaselines3.drl_agent import DRLAgent
from finrl.plot import backtest_stats, backtest_plot, get_daily_return, get_baseline

import pandas as pd

MODELS = {"A2C": A2C, "DDPG": DDPG, "TD3": TD3, "SAC": SAC, "PPO": PPO}


class StocksStrategy:

    def __init__(self, model_name, root_dir, results_dir, trained_model_dir, prefix='stocks'):
        self.prefix = prefix
        self.root_dir = root_dir
        self.results_dir = results_dir
        self.trained_model_dir = trained_model_dir
        self.model_name = model_name
        self.results_account_value = None

        self.results_file_prefix = f"{self.root_dir}/{self.results_dir}/{self.prefix}_{model_name}"

        if self.model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")

    def train(self, env, timesteps, use_existing=False):
        trained = self.train_model(DRLAgent(env=env), timesteps, use_existing)
        self.save_model(trained)

    def train_model(self, agent, timesteps, use_existing=False):
        model = self.get_model(agent, use_trained_if_available=use_existing)
        trained_model = agent.train_model(model=model, tb_log_name=self.model_name, total_timesteps=timesteps)
        return trained_model

    def get_model_type(self):
        model_type = None
        if self.model_name == "A2C":
            model_type = A2C
        if self.model_name == "DDPG":
            model_type = DDPG
        if self.model_name == "PPO":
            model_type = PPO
        if self.model_name == "SAC":
            model_type = SAC
        if self.model_name == "TD3":
            model_type = TD3
        return model_type

    def get_model_filename(self):
        return f"{self.root_dir}/{self.trained_model_dir}/{self.prefix}_{self.model_name}_MODEL"

    def get_model(self, agent, use_trained_if_available=False):
        model = None
        if use_trained_if_available and self.model_file_exists():
            model = self.load_model()
            model.set_env(agent.env)
        if model is None:
            model = self.create_model(agent)
        return model

    def model_file_exists(self):
        return os.path.isfile(f"{self.get_model_filename()}.zip")

    def load_model(self):
        """
        Loads a saved model file from file system
        :return: stored model from disk
        """
        if not self.model_file_exists():
            raise ValueError("NoModelFileAvailableError")
        model_type = self.get_model_type()
        model = model_type.load(f"{self.get_model_filename()}")
        return model

    def save_model(self, model):
        model.save(f"{self.get_model_filename()}")

    def create_model(self, agent):
        """
        Creates a new model based on the algorithm name provided
        :param agent: DRL Agent
        :return: new created model
        """
        model = None
        if self.model_name not in MODELS:
            raise NotImplementedError("NotImplementedError")
        if self.model_name == "A2C":
            model = agent.get_model("A2C")
        if self.model_name == "DDPG":
            model = agent.get_model("DDPG")
        if self.model_name == "PPO":
            ppo_params = {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 128}
            model = agent.get_model("PPO", model_kwargs=ppo_params)
        if self.model_name == "TD3":
            td3_params = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
            model = agent.get_model("TD3", model_kwargs=td3_params)
        if self.model_name == "SAC":
            sac_params = {"batch_size": 128, "buffer_size": 1000000, "learning_rate": 0.0001,
                          "learning_starts": 100, "ent_coef": "auto_0.1"}
            model = agent.get_model("SAC", model_kwargs=sac_params)
        return model

    def test(self, env):
        loaded_model = self.load_model()
        df_account_value, df_actions = DRLAgent.DRL_prediction(model=loaded_model, environment=env)

        result_prefix = f"{self.root_dir}/{self.results_dir}/{self.prefix}_{self.model_name}"
        df_account_value.to_csv(f"{result_prefix}_portfolio_value.csv")
        df_actions.to_csv(f"{result_prefix}_portfolio_actions.csv")

    def generate_backtest_results(self):
        print("==============Get Backtest Results===========")
        perf_stats_all = backtest_stats(account_value=self.get_results_account_value())
        perf_stats_all = pd.DataFrame(perf_stats_all)
        perf_stats_all.to_csv(f"{self.results_file_prefix}_perf_stats.csv")

    def generate_baseline_stats(self):
        print("==============Get Baseline Stats===========")
        acc_val = self.get_results_account_value()

        baseline = get_baseline(ticker="^DJI", start=acc_val.loc[0, 'date'],
                                end=acc_val.loc[len(acc_val) - 1, 'date'])
        df2 = acc_val.copy()
        df2["date"] = pd.to_datetime(df2["date"])
        baseline["date"] = pd.to_datetime(baseline["date"], format="%Y-%m-%d")
        baseline = pd.merge(df2[["date"]], baseline, how="left", on="date")
        baseline = baseline.fillna(method="ffill").fillna(method="bfill")
        baseline_returns = get_daily_return(baseline, value_col_name="close")
        baseline_returns.to_csv(f"{self.results_file_prefix}_base_returns.csv")
        baseline_stats = backtest_stats(baseline, value_col_name='close')
        baseline_stats = pd.DataFrame(baseline_stats)
        baseline_stats.to_csv(f"{self.results_file_prefix}_base_stats.csv")

    def get_results_account_value(self):
        if self.results_account_value is None:
            filename = f"{self.results_file_prefix}_portfolio_value.csv"
            self.results_account_value = pd.read_csv(filename, index_col=0)
        return self.results_account_value

    def backtest_plot(self):
        print("==============Compare to DJIA===========")
        df = self.get_results_account_value()
        a = backtest_plot(df,
                          baseline_ticker='^DJI',
                          baseline_start=df.loc[0, 'date'],
                          baseline_end=df.loc[len(df) - 1, 'date'])
