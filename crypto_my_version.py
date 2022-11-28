import pandas as pd

from config import crypto
from config import general as config
from lib.drl import data_split, train, test, get_model_params
from lib.support import check_run_directory_structure, get_run_timestamp

# global settings
ROOT_DIR = '.'
# check_directory_structure(ROOT_DIR)
STRATEGY_NAME = "cs"
MODEL_DIR = f"{ROOT_DIR}/{config.TRAINED_MODEL_DIR}/{STRATEGY_NAME}"
TENSORBOARD_DIR = f"./tensorboard_log/{STRATEGY_NAME}"


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

# Settings
MODEL_NAME = "A2C"
model_params = get_model_params(MODEL_NAME)
RUN_NAME = get_run_timestamp() + "_CFG1_10M_debug"

print(f"Using Model {MODEL_NAME} as {RUN_NAME} with params={model_params}")
check_run_directory_structure(ROOT_DIR, config.RESULTS_DIR, STRATEGY_NAME, MODEL_NAME, RUN_NAME)

results_file_prefix = f"{ROOT_DIR}/{config.RESULTS_DIR}/{STRATEGY_NAME}/{MODEL_NAME}/{MODEL_NAME}_{RUN_NAME}"
model_filename = f"{MODEL_DIR}/{STRATEGY_NAME}_{MODEL_NAME}_{RUN_NAME}"

retrain_existing_model = True
previous_model_name = f"./trained_models/cs/cs_A2C_11272204_CFG1_10M"

ENV_KWARGS['run_name'] = RUN_NAME
ENV_KWARGS['model_name'] = MODEL_NAME
timesteps = 1_000_000

settings = {
    "total_timesteps": timesteps,
    "retrain_existing_model": retrain_existing_model,
    "previous_model_name": previous_model_name,
    "tensorboard_log": TENSORBOARD_DIR,
    "env_kwargs": ENV_KWARGS,
    "model_params": model_params,
    "save_model": True,
    "target_model_filename": model_filename,
    "file_prefix": results_file_prefix
}

# ===== TRAIN
trained = train(train_df, ENV_KWARGS, settings)

# ===== TEST
test(test_df, ENV_KWARGS, settings)
