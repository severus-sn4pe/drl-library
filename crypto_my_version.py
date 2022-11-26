import os
import time

import pandas as pd
import datetime

from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3

from config import crypto
from config import general as config
from finrl.agents.stablebaselines3.drl_agent import DRLAgent
from finrl.meta.env_custom.env_custom import CustomTradingEnv
from lib.drl import data_split
from lib.support import log_duration

ROOT_DIR = '.'
# check_directory_structure(ROOT_DIR)

STRATEGY_NAME = "cs"
RUN_NAME = datetime.datetime.now().strftime("%Y%m%d_%H%M")
MODEL_NAME = "A2C"

print(f"RUN_NAME={RUN_NAME}")
TENSORBOARD_DIR = f"./tensorboard_log/{STRATEGY_NAME}"
file_start = time.time()

MODELS = {"A2C": A2C, "DDPG": DDPG, "TD3": TD3, "SAC": SAC, "PPO": PPO}

if not os.path.exists(f"{ROOT_DIR}/{config.RESULTS_DIR}/{STRATEGY_NAME}"):
    os.mkdir(f"{ROOT_DIR}/{config.RESULTS_DIR}/{STRATEGY_NAME}")
if not os.path.exists(f"{ROOT_DIR}/{config.RESULTS_DIR}/{STRATEGY_NAME}/{MODEL_NAME}"):
    os.mkdir(f"{ROOT_DIR}/{config.RESULTS_DIR}/{STRATEGY_NAME}/{MODEL_NAME}")
if not os.path.exists(f"{ROOT_DIR}/{config.RESULTS_DIR}/{STRATEGY_NAME}/{MODEL_NAME}/{RUN_NAME}"):
    os.mkdir(f"{ROOT_DIR}/{config.RESULTS_DIR}/{STRATEGY_NAME}/{MODEL_NAME}/{RUN_NAME}")


def get_model(a, model_name):
    m = None
    if model_name not in MODELS:
        raise NotImplementedError("NotImplementedError")
    if model_name == "A2C":
        m = a.get_model("A2C", tensorboard_log=f'{TENSORBOARD_DIR}')
    if model_name == "DDPG":
        m = a.get_model("DDPG", tensorboard_log=f'{TENSORBOARD_DIR}')
    if model_name == "PPO":
        ppo_params = {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 128}
        m = a.get_model("PPO", model_kwargs=ppo_params, tensorboard_log=f'{TENSORBOARD_DIR}')
    if model_name == "TD3":
        td3_params = {"batch_size": 100, "buffer_size": 1000000, "learning_rate": 0.001}
        m = a.get_model("TD3", model_kwargs=td3_params, tensorboard_log=f'{TENSORBOARD_DIR}')
    if model_name == "SAC":
        sac_params = {"batch_size": 128, "buffer_size": 1000000, "learning_rate": 0.0001, "learning_starts": 100,
                      "ent_coef": "auto_0.1"}
        m = a.get_model("SAC", model_kwargs=sac_params, tensorboard_log=f'{TENSORBOARD_DIR}')
    return m


def get_train_env(df_tr, name):
    kwargs = ENV_KWARGS.copy()
    kwargs['model_name'] = name
    e_train_gym = CustomTradingEnv(df=df_tr, **kwargs)
    env_train, _ = e_train_gym.get_sb_env()
    return env_train


def get_test_env(df_tst, turb_thres=None):
    kwargs = ENV_KWARGS.copy()
    kwargs['mode'] = 'test'
    e_trade_gym = CustomTradingEnv(df=df_tst, turbulence_threshold=turb_thres, **kwargs)
    return e_trade_gym


def load_model_from_file(model_name, filename):
    model_file_exists = os.path.isfile(f"{filename}.zip")
    if not model_file_exists:
        raise ValueError("NoModelFileAvailableError")

    model_type = MODELS[model_name]
    loaded_model = model_type.load(f"{filename}.zip", tensorboard_log=TENSORBOARD_DIR)
    print(f"loaded model from {filename}")
    return loaded_model


df = pd.read_csv(f"{config.DATA_SAVE_DIR}/thesis/crypto_1d_parsed.csv", index_col=0)
train_df = data_split(df, crypto.TRAIN_START_DATE, crypto.TRAIN_END_DATE)
test_df = data_split(df, crypto.TEST_START_DATE, crypto.TEST_END_DATE)
print(f"train {train_df.shape} start: {crypto.TRAIN_START_DATE} end: {crypto.TRAIN_END_DATE}")
print(f"test  {test_df.shape} start: {crypto.TEST_START_DATE} end: {crypto.TEST_END_DATE}")

stock_dimension = len(train_df.tic.unique())
state_space = 1 + 2 * stock_dimension + len(crypto.INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

ENV_KWARGS = {
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
    "mode": "train",
    "strategy_name": STRATEGY_NAME,
    "run_name": RUN_NAME
}


results_file_prefix = f"{ROOT_DIR}/{config.RESULTS_DIR}/{STRATEGY_NAME}/{MODEL_NAME}/{MODEL_NAME}_{RUN_NAME}"
model_filename = f"{ROOT_DIR}/{config.TRAINED_MODEL_DIR}/{STRATEGY_NAME}/{STRATEGY_NAME}_{MODEL_NAME}_{RUN_NAME}"

env_train = get_train_env(train_df, MODEL_NAME)

####  ===== TRAIN

# # TRAIN
total_timesteps = 2_000_000
agent = DRLAgent(env=env_train)

USE_EXISTING_MODEL = True

if USE_EXISTING_MODEL:
    previous_model_name = f"{ROOT_DIR}/{config.TRAINED_MODEL_DIR}/{STRATEGY_NAME}/{STRATEGY_NAME}_{MODEL_NAME}_20221126_0304_20M"
    model = load_model_from_file(MODEL_NAME, previous_model_name)
    model.set_env(env_train)
else:
    # initialize new model
    model = get_model(agent, MODEL_NAME)

start = time.time()
trained_model = agent.train_model(model=model, tb_log_name=f"{MODEL_NAME}_{RUN_NAME}", total_timesteps=total_timesteps)
log_duration(start)

trained_model.save(model_filename)
print(f"Storing model in {model_filename}")
####  ===== TEST


env_test = get_test_env(test_df)
model = load_model_from_file(MODEL_NAME, model_filename)

start = time.time()
df_account_value, df_actions = DRLAgent.DRL_prediction(model=model, environment=env_test)
log_duration(start)

df_account_value.to_csv(f"{results_file_prefix}_portfolio_value.csv")
df_actions.to_csv(f"{results_file_prefix}_actions.csv")
