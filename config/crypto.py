# date format: '%Y-%m-%d'
# TOTAL_START_DATE = "2009-01-01"
# # TOTAL_START_DATE = "2014-01-01"
# TOTAL_END_DATE = "2022-07-01"

TRAIN_START_DATE = "2021-05-11"  # 00:00:00"
# TRAIN_START_DATE = "2021-11-15 00:00:00"
# TRAIN_START_DATE = "2022-01-24 00:00:00"

TRAIN_END_DATE = "2022-05-24 23:59:59"

TEST_START_DATE = "2022-05-25"  # 00:00:00"
TEST_END_DATE = "2022-11-01 23:59:59"

VALIDATION_ITERATIONS = 7
#
# TRADE_START_DATE = "2022-01-01"
# TRADE_END_DATE = "2022-07-01"

# stockstats technical indicator column names
# check https://pypi.org/project/stockstats/ for different names
# INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]
INDICATORS = ["macd", "rsi_30", "cci_30", "dx_30"]
INDICATORS_PLUS = ['macd', 'boll_ub', 'boll_lb', 'rsi_30', 'cci_30', 'dx_30', 'close_7_sma', 'close_30_sma']

# Default Model Parameters
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007, "device": "cpu"}
PPO_PARAMS = {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 64, "device": "cpu"}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50_000, "learning_rate": 0.001, "device": "cpu"}
TD3_PARAMS = {"batch_size": 128, "buffer_size": 50_000, "learning_rate": 0.001, "tau": 0.005, "device": "cpu"}
SAC_PARAMS = {"batch_size": 128, "buffer_size": 100_000,
              "learning_rate": 0.0001, "learning_starts": 100,
              "ent_coef": "auto_0.1", "device": "cpu"}

MODEL_PARAMS = {
    "A2C": {
        "default": {"n_steps": 512, "ent_coef": 1e-7, "learning_rate": 1e-3, "device": "cpu"},
        "V212": {"n_steps": 512, "ent_coef": 1e-7, "learning_rate": 1e-3, "device": "cpu"},
        "V213": {"n_steps": 1024, "ent_coef": 1e-7, "learning_rate": 1e-3, "device": "cpu"},
        "V218": {"n_steps": 512, "ent_coef": 1e-6, "learning_rate": 1e-3, "device": "cpu"},
        "V219": {"n_steps": 1024, "ent_coef": 1e-6, "learning_rate": 1e-3, "device": "cpu"},
        "V220": {"n_steps": 512, "ent_coef": 1e-5, "learning_rate": 1e-3, "device": "cpu"},
        "V221": {"n_steps": 1024, "ent_coef": 1e-5, "learning_rate": 1e-3, "device": "cpu"},
    },
    "PPO": {
        "default": {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 128, "device": "cpu"},
        "V201": {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 128, "device": "cpu"},
        "V202": {"n_steps": 1024, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 128, "device": "cpu"},
        "V203": {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 1024, "device": "cpu"},
        "V204": {"n_steps": 1024, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 1024, "device": "cpu"},
        "V205": {"n_steps": 2048, "ent_coef": 0.001, "learning_rate": 0.00025, "batch_size": 1024, "device": "cpu"},
        "V206": {"n_steps": 1024, "ent_coef": 0.001, "learning_rate": 0.00025, "batch_size": 1024, "device": "cpu"},
        "V207": {"n_steps": 4096, "ent_coef": 0.001, "learning_rate": 0.00025, "batch_size": 1024, "device": "cpu"},
        "V208": {"n_steps": 2048, "ent_coef": 0.001, "learning_rate": 0.001, "batch_size": 1024, "device": "cpu"},
    },
    "TD3": {
        "default": {"batch_size": 32, "buffer_size": 50_000, "learning_rate": 0.001, "tau": 0.005, "device": "cpu"},
        "V201": {"batch_size": 32, "buffer_size": 50_000, "learning_rate": 0.001, "tau": 0.005, "device": "cpu"},
        "V202": {"batch_size": 128, "buffer_size": 50_000, "learning_rate": 0.001, "tau": 0.005, "device": "cpu"},
        "V203": {"batch_size": 1024, "buffer_size": 50_000, "learning_rate": 0.001, "tau": 0.005, "device": "cpu"},
        "V204": {"batch_size": 32, "buffer_size": 50_000, "learning_rate": 0.001, "tau": 0.05, "device": "cpu"},
        "V205": {"batch_size": 128, "buffer_size": 50_000, "learning_rate": 0.001, "tau": 0.05, "device": "cpu"},
        "V206": {"batch_size": 1024, "buffer_size": 50_000, "learning_rate": 0.001, "tau": 0.05, "device": "cpu"},
        "V207": {"batch_size": 32, "buffer_size": 50_000, "learning_rate": 0.01, "tau": 0.005, "device": "cpu"},
        "V208": {"batch_size": 128, "buffer_size": 50_000, "learning_rate": 0.01, "tau": 0.005, "device": "cpu"},
        "V209": {"batch_size": 1024, "buffer_size": 50_000, "learning_rate": 0.01, "tau": 0.005, "device": "cpu"},
        "V210": {"batch_size": 32, "buffer_size": 50_000, "learning_rate": 0.01, "tau": 0.05, "device": "cpu"},
        "V211": {"batch_size": 128, "buffer_size": 50_000, "learning_rate": 0.01, "tau": 0.05, "device": "cpu"},
        "V212": {"batch_size": 1024, "buffer_size": 50_000, "learning_rate": 0.01, "tau": 0.05, "device": "cpu"},
    },
    "DDPG": {
        "default": {"batch_size": 128, "buffer_size": 50000, "learning_rate": 0.001, "device": "cpu"}
    },
    "SAC": {
        "default": {
            "batch_size": 128, "buffer_size": 100_000,
            "learning_rate": 0.0001, "learning_starts": 100,
            "ent_coef": "auto_0.1", "device": "cpu"
        }
    }
}
