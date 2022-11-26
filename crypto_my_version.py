import datetime
import time

import pandas as pd

from config import crypto
from config import general as config
from finrl.agents.stablebaselines3.drl_agent import DRLAgent
from lib.drl import data_split, get_train_env, get_test_env, load_model_from_file
from lib.support import log_duration, check_run_directory_structure

# global settings
ROOT_DIR = '.'
# check_directory_structure(ROOT_DIR)
STRATEGY_NAME = "cs"
MODEL_DIR = f"{ROOT_DIR}/{config.TRAINED_MODEL_DIR}/{STRATEGY_NAME}"
TENSORBOARD_DIR = f"./tensorboard_log/{STRATEGY_NAME}"


def get_model_params(model_name):
    params = {}
    if model_name == "A2C":
        params = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
    if model_name == "DDPG":
        params = {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001}
    if model_name == "PPO":
        params = {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 128}
    if model_name == "TD3":
        params = {"batch_size": 100, "buffer_size": 1_000_000, "learning_rate": 0.001}
    if model_name == "SAC":
        params = {
            "batch_size": 128, "buffer_size": 100_000,
            "learning_rate": 0.0001, "learning_starts": 100, "ent_coef": "auto_0.1"}
    return params


# loading dataset
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
    "hmax": 10_000,
    "initial_amount": 1_000_000,
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
    "run_name": "PLACEHOLDER",
    "model_name": "PLACEHOLDER"
}

file_start = time.time()

# Run Settings
RUN_NAME = datetime.datetime.now().strftime("%Y%m%d_%H%M") + "_test"
MODEL_NAME = "A2C"
model_params = get_model_params(MODEL_NAME)
print(f"Using Model {MODEL_NAME} as {RUN_NAME} with params={model_params}")
check_run_directory_structure(ROOT_DIR, config.RESULTS_DIR, STRATEGY_NAME, MODEL_NAME, RUN_NAME)

results_file_prefix = f"{ROOT_DIR}/{config.RESULTS_DIR}/{STRATEGY_NAME}/{MODEL_NAME}/{MODEL_NAME}_{RUN_NAME}"
model_filename = f"{MODEL_DIR}/{STRATEGY_NAME}_{MODEL_NAME}_{RUN_NAME}"

# ===== TRAIN
total_timesteps = 2_000_0

ENV_KWARGS['run_name'] = RUN_NAME
ENV_KWARGS['model_name'] = MODEL_NAME
env_train = get_train_env(train_df, ENV_KWARGS)
agent = DRLAgent(env=env_train)

USE_EXISTING_MODEL = True

if USE_EXISTING_MODEL:
    previous_model_name = f"{MODEL_DIR}/{STRATEGY_NAME}_{MODEL_NAME}_20221126_0304_20M"
    model = load_model_from_file(MODEL_NAME, previous_model_name, TENSORBOARD_DIR)
    model.set_env(env_train)
else:
    # initialize new model
    model = agent.get_model(MODEL_NAME, model_kwargs=model_params, tensorboard_log=TENSORBOARD_DIR)

start = time.time()
trained_model = agent.train_model(model=model, tb_log_name=f"{MODEL_NAME}_{RUN_NAME}", total_timesteps=total_timesteps)
log_duration(start)

trained_model.save(model_filename)
print(f"Storing model in {model_filename}")

# ===== TEST
env_test = get_test_env(test_df, MODEL_NAME, ENV_KWARGS)
model = load_model_from_file(MODEL_NAME, model_filename, TENSORBOARD_DIR)

start = time.time()
df_account_value, df_actions = DRLAgent.DRL_prediction(model=model, environment=env_test)
log_duration(start)

df_account_value.to_csv(f"{results_file_prefix}_portfolio_value.csv")
df_actions.to_csv(f"{results_file_prefix}_actions.csv")
