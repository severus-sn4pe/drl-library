import datetime

import pandas as pd

from config import general as config
from config import stocks
from finrl.agents.stablebaselines3.drl_ensemble_agent import DRLEnsembleAgent
from finrl.plot import backtest_stats, backtest_plot, get_baseline
from lib.drl import load_dataset
from lib.support import check_directory_structure

FILE_PREFIX = "stocks"
ROOT_DIR = '..'

check_directory_structure(ROOT_DIR)
processed_full = load_dataset(f'{ROOT_DIR}/datasets/stocks/DOW30.csv', stocks.INDICATORS_LIGHT, use_turbulence=True,
                              use_vix=False)

stock_dimension = len(processed_full.tic.unique())
tech_indicators = stocks.INDICATORS_LIGHT

state_space = 1 + 2 * stock_dimension + len(tech_indicators) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "buy_cost_pct": 0.001,
    "sell_cost_pct": 0.001,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": tech_indicators,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4,
    "print_verbosity": 5
}

# rebalance_window is the number of days to retrain the model
rebalance_window = 63

# validation_window is the number of days to do validation and trading (e.g. if validation_window=63,
# then both validation and trading period will be 63 days)
validation_window = 63

# train_start = '2009-04-01'
train_start = '2018-04-01'
# train_end = '2021-04-01'
train_end = '2021-11-01'
# val_test_start = '2021-04-01'
val_test_start = '2021-11-01'
val_test_end = '2022-07-01'

ensemble_agent = DRLEnsembleAgent(df=processed_full,
                                  train_period=(train_start, train_end),
                                  val_test_period=(val_test_start, val_test_end),
                                  rebalance_window=rebalance_window,
                                  validation_window=validation_window,
                                  root_dir=ROOT_DIR,
                                  trained_model_dir=config.TRAINED_MODEL_DIR,
                                  results_dir=config.RESULTS_DIR,
                                  tensorboard_log_dir=config.TENSORBOARD_LOG_DIR,
                                  **env_kwargs)

A2C_model_kwargs = {'n_steps': 5, 'ent_coef': 0.01, 'learning_rate': 0.0005}
PPO_model_kwargs = {"ent_coef": 0.01, "n_steps": 2048, "learning_rate": 0.00025, "batch_size": 64}
DDPG_model_kwargs = {"buffer_size": 10_000, "learning_rate": 0.0005, "batch_size": 64}
timesteps_dict = {'A2C': 25, 'PPO': 25, 'DDPG': 25}
# timesteps_dict = {'A2C': 10_000, 'PPO': 10_000, 'DDPG': 10_000}

df_summary = ensemble_agent.run_ensemble_strategy(A2C_model_kwargs,
                                                  PPO_model_kwargs,
                                                  DDPG_model_kwargs,
                                                  timesteps_dict)

# Part 7 - backtest strategy

unique_trade_date = processed_full[
    (processed_full.date > val_test_start) & (processed_full.date <= val_test_end)].date.unique()

df_trade_date = pd.DataFrame({'datadate': unique_trade_date})

df_account_value = pd.DataFrame()
for i in range(rebalance_window + validation_window, len(unique_trade_date) + 1, rebalance_window):
    temp = pd.read_csv(f"{ROOT_DIR}/results/account_value_trade_ensemble_{i}.csv")
    df_account_value = df_account_value.append(temp, ignore_index=True)
sharpe = (252 ** 0.5) * df_account_value.account_value.pct_change(1).mean() / df_account_value.account_value.pct_change(
    1).std()
print('Sharpe Ratio: ', sharpe)
df_account_value = df_account_value.join(df_trade_date[validation_window:].reset_index(drop=True))

# %matplotlib inline
# df_account_value.account_value.plot()


print("==============Get Backtest Results===========")
now = datetime.datetime.now().strftime('%Y%m%d-%Hh%M')

perf_stats_all = backtest_stats(account_value=df_account_value)
perf_stats_all = pd.DataFrame(perf_stats_all)

# baseline stats
print("==============Get Baseline Stats===========")
baseline_df = get_baseline(
    ticker="^DJI",
    start=df_account_value.loc[0, 'date'],
    end=df_account_value.loc[len(df_account_value) - 1, 'date'])

stats = backtest_stats(baseline_df, value_col_name='close')

print("==============Compare to DJIA===========")
# %matplotlib inline
# S&P 500: ^GSPC
# Dow Jones Index: ^DJI
# NASDAQ 100: ^NDX
backtest_plot(df_account_value,
              baseline_ticker='^DJI',
              baseline_start=df_account_value.loc[0, 'date'],
              baseline_end=df_account_value.loc[len(df_account_value) - 1, 'date'])

print("done")
