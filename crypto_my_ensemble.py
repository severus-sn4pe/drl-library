import pandas as pd

from config import crypto
from config import general as config
from finrl.agents.stablebaselines3.drl_ensemble_agent import DRLEnsembleAgent
from lib.support import get_run_timestamp

# global settings
ROOT_DIR = '.'
# check_directory_structure(ROOT_DIR)
STRATEGY_NAME = "ce"  # Crypto Ensemble Strategy
RES = "1d"
RUN_NAME = f"{STRATEGY_NAME}_{RES}_{get_run_timestamp()}"

MODEL_DIR = f"{ROOT_DIR}/{config.TRAINED_MODEL_DIR}/{STRATEGY_NAME}"
TENSORBOARD_DIR = f"./tensorboard_log/{STRATEGY_NAME}"

MODEL_SETTINGS = {
    "models": ["A2C", "PPO", "TD3"],
    "A2C": {
        "model_name": "A2C",
        "params": {
            "n_steps": 1024,
            "ent_coef": 1e-5,
            "learning_rate": 1e-3,
            "device": "cpu"},
        "init_model": "final_trained/A2C_1D_260M",
        # "init_model": "final_trained/A2C_12H_600M",
        # "init_model": "final_trained/A2C_6H_550M",
        # "init_model": "final_trained/A2C_1H_240M",
        "timesteps": 75800 * 6
        # "timesteps": 90960
    },
    "PPO": {
        "model_name": "PPO",
        "params": {
            "n_steps": 2048,
            "ent_coef": 0.001,
            "learning_rate": 0.001,
            "batch_size": 1024,
            "device": "cpu"},
        "init_model": "final_trained/PPO_1D_180M",
        # "init_model": "final_trained/PPO_12H_600M",
        # "init_model": "final_trained/PPO_6H_550M",
        # "init_model": "final_trained/PPO_1H_200M",
        "timesteps": 75800 * 6
        # "timesteps": 90960
    },
    "TD3": {
        "model_name": "TD3",
        "params": {
            "batch_size": 2048,
            "buffer_size": 50_000,
            "learning_rate": 0.001,
            "tau": 0.05,
            "device": "cuda"
        },
        "init_model": "final_trained/TD3_1D_18M",
        # "init_model": "final_trained/TD3_12H_54M",
        # "init_model": "final_trained/TD3_6H_54M",
        # "init_model": "final_trained/TD3_1H_42M",
        "timesteps": 37900 * 6
        # "timesteps": 45480
    },
}

df = pd.read_csv(f"{config.DATA_SAVE_DIR}/thesis/crypto_{RES}_plus.csv", index_col=0)
stock_dimension = len(df.tic.unique())
num_stock_shares = [0] * stock_dimension

# check_run_directory_structure(ROOT_DIR, config.RESULTS_DIR, STRATEGY_NAME, MODEL_NAME, RUN_NAME)

AGENT_KWARGS = {
    "strategy_name": STRATEGY_NAME,
    "run_name": RUN_NAME,
    "stock_dim": stock_dimension,
    "initial_amount": 1_000_000,
    "initial_num_stock_shares": num_stock_shares,
    "buy_cost_pct": crypto.FEE_PERCENTAGE,
    "sell_cost_pct": crypto.FEE_PERCENTAGE,
    "reward_scaling": 1e-6,
    "tech_indicator_list": crypto.INDICATORS_PLUS,
    "print_verbosity": 1e7,
    "make_plots": True,
    "model_settings": MODEL_SETTINGS,
    "iterations": crypto.ENSEMBLE_ITERATIONS,
    "start_dates": {
        "train": crypto.ENSEMBLE_TRAIN_START,
        "val": crypto.ENSEMBLE_VAL_START,
        "test": crypto.ENSEMBLE_TEST_START,
    },
    "windows": {
        "train": crypto.ENSEMBLE_TRAIN_WINDOW,
        "val": crypto.ENSEMBLE_VALIDATION_WINDOW,
    },
    "res": RES,
    "root_dir": ROOT_DIR,
    "trained_model_dir": config.TRAINED_MODEL_DIR,
    "results_dir": config.RESULTS_DIR,
    "tensorboard_log_dir": config.TENSORBOARD_LOG_DIR,

}
if __name__ == "__main__":
    print(f"Running {STRATEGY_NAME} with name {RUN_NAME}")
    print(f"=============================================================")
    ensemble_agent = DRLEnsembleAgent(df=df, **AGENT_KWARGS)
    ensemble_agent.run_ensemble()
