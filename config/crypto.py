# date format: '%Y-%m-%d'
# TOTAL_START_DATE = "2009-01-01"
# # TOTAL_START_DATE = "2014-01-01"
# TOTAL_END_DATE = "2022-07-01"

TRAIN_START_DATE = "2021-05-11 00:00:00"
# TRAIN_START_DATE = "2021-11-15 00:00:00"
# TRAIN_START_DATE = "2022-01-24 00:00:00"

TRAIN_END_DATE = "2022-05-24 23:59:59"

TEST_START_DATE = "2022-05-25 00:00:00"
TEST_END_DATE = "2022-11-01 23:59:59"
#
# TRADE_START_DATE = "2022-01-01"
# TRADE_END_DATE = "2022-07-01"

# stockstats technical indicator column names
# check https://pypi.org/project/stockstats/ for different names
# INDICATORS = ["macd", "boll_ub", "boll_lb", "rsi_30", "cci_30", "dx_30", "close_30_sma", "close_60_sma"]
INDICATORS = ["macd", "rsi_30", "cci_30", "dx_30"]

# Model Parameters
A2C_PARAMS = {"n_steps": 5, "ent_coef": 0.01, "learning_rate": 0.0007}
PPO_PARAMS = {"n_steps": 2048, "ent_coef": 0.01, "learning_rate": 0.00025, "batch_size": 64}
DDPG_PARAMS = {"batch_size": 128, "buffer_size": 50_000, "learning_rate": 0.001}
TD3_PARAMS = {"batch_size": 100, "buffer_size": 1_000_000, "learning_rate": 0.001}
SAC_PARAMS = {"batch_size": 64, "buffer_size": 100_000, "learning_rate": 0.0001, "learning_starts": 100,
              "ent_coef": "auto_0.1"}
