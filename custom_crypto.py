from config import crypto
from config import general as config
from finrl.meta.env_cryptocurrency_trading.env_multiple_crypto import CryptoEnv
from finrl.meta.env_custom.env_custom import CustomTradingEnv
from lib.drl import load_dataset, data_split
from lib.stocks_strategy import StocksStrategy
from lib.support import check_directory_structure
# from finrl.agents.elegantrl.elegantrl_models import DRLAgent as DRLAgent_erl
from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3

import numpy as np
import pandas as pd
import time
import os

from finrl.agents.stablebaselines3.drl_agent import DRLAgent
from finrl.meta.data_processor import DataProcessor

file_start = time.time()

ROOT_DIR = '.'
# check_directory_structure(ROOT_DIR)

FILE_PREFIX = "BTCUSDT"

MODELS = {"A2C": A2C, "DDPG": DDPG, "TD3": TD3, "SAC": SAC, "PPO": PPO}

# FILE_PREFIX = 'crypto10'
df = pd.read_csv(f"{config.DATA_SAVE_DIR}/crypto/{FILE_PREFIX}_1h_parsed.csv", index_col=0)
train_df = data_split(df, crypto.TRAIN_START_DATE, crypto.TRAIN_END_DATE)
test_df = data_split(df, crypto.TEST_START_DATE, crypto.TEST_END_DATE)
print(f"train {train_df.shape} start: {crypto.TRAIN_START_DATE} end: {crypto.TRAIN_END_DATE}")
print(f"test  {test_df.shape} start: {crypto.TEST_START_DATE} end: {crypto.TEST_END_DATE}")

stock_dimension = len(train_df.tic.unique())
state_space = 1 + 2 * stock_dimension + len(crypto.INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension # cost per stock, we use the same for all, but could be varying
num_stock_shares = [0] * stock_dimension # how many stocks are in portfolio at the begin of the training, we initialize all with 0 for an empty portfolio

env_kwargs = {
    "hmax": 10000,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": crypto.INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
    "make_plots": True,
    "mode": "train"
}
e_train_gym = CustomTradingEnv(df=train_df, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()

model_name="PPO"
env_kwargs['model_name'] = model_name
prefix = "crypto"
root_dir=ROOT_DIR
results_dir=config.RESULTS_DIR
trained_model_dir=config.TRAINED_MODEL_DIR
results_file_prefix = f"{root_dir}/{results_dir}/{prefix}_{model_name}"
if model_name not in MODELS:
    raise NotImplementedError("NotImplementedError")

# # TRAIN
total_timesteps = 1000
agent = DRLAgent(env=env_train)

model = None
if model_name == "A2C":
    model = agent.get_model("A2C")
if model_name == "DDPG":
    model = agent.get_model("DDPG")
if model_name == "PPO":
    ppo_params = {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 128}
    model = agent.get_model("PPO", model_kwargs=ppo_params)
if model_name == "TD3":
    td3_params = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
    model = agent.get_model("TD3", model_kwargs=td3_params)
if model_name == "SAC":
    sac_params = {"batch_size": 128, "buffer_size": 1000000, "learning_rate": 0.0001, "learning_starts": 100, "ent_coef": "auto_0.1"}
    model = agent.get_model("SAC", model_kwargs=sac_params)

start = time.time()
trained_model = agent.train_model(model=model, tb_log_name=model_name, total_timesteps=total_timesteps)
log_duration(start)

model_filename = f"{root_dir}/{trained_model_dir}/{prefix}_{model_name}_MODEL"
trained_model.save(model_filename)

env_kwargs['mode'] = 'test'
e_trade_gym = CustomTradingEnv(df=test_df, turbulence_threshold=None, **env_kwargs)

model_file_exists = os.path.isfile(f"{model_filename}.zip")
if not model_file_exists:
    raise ValueError("NoModelFileAvailableError")

model_type = MODELS[model_name]
loaded_model = model_type.load(f"{model_filename}.zip")

start = time.time()
df_account_value, df_actions = DRLAgent.DRL_prediction(model=loaded_model, environment=e_trade_gym)
log_duration(start)

df_account_value.to_csv(f"{results_file_prefix}_portfolio_value.csv")
df_actions.to_csv(f"{results_file_prefix}_portfolio_actions.csv")

print("done")
log_duration(file_start)
