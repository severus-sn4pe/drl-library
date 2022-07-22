from config import stocks
from config import general as config
from finrl.meta.env_stock_trading.env_stocktrading import StockTradingEnv
from lib.drl import load_dataset, data_split
from lib.stocks import StocksStrategy
from lib.support import check_directory_structure

ROOT_DIR = '..'
check_directory_structure(ROOT_DIR)

FILE_PREFIX = "stocks"

df = load_dataset(f'{ROOT_DIR}/datasets/stocks/DOW30.csv', stocks.INDICATORS, use_turbulence=True, use_vix=True)

train_df = data_split(df, '2009-01-01', '2020-07-01')
trade_df = data_split(df, '2020-07-01', '2022-07-01')

print(f"Train Shape: {train_df.shape}")
print(f"Test Shape: {trade_df.shape}")

stock_dimension = len(train_df.tic.unique())
state_space = 1 + 2 * stock_dimension + len(stocks.INDICATORS) * stock_dimension
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")

buy_cost_list = sell_cost_list = [0.001] * stock_dimension
num_stock_shares = [0] * stock_dimension

env_kwargs = {
    "hmax": 100,
    "initial_amount": 1000000,
    "num_stock_shares": num_stock_shares,
    "buy_cost_pct": buy_cost_list,
    "sell_cost_pct": sell_cost_list,
    "state_space": state_space,
    "stock_dim": stock_dimension,
    "tech_indicator_list": stocks.INDICATORS,
    "action_space": stock_dimension,
    "reward_scaling": 1e-4
}

e_train_gym = StockTradingEnv(df=train_df, **env_kwargs)
try:
    env_train, _ = e_train_gym.get_sb_env()
except ValueError:
    env_train, _ = e_train_gym.get_sb_env()

stockStrategy = StocksStrategy(model_name="SAC",
                               root_dir=ROOT_DIR,
                               results_dir=config.RESULTS_DIR,
                               trained_model_dir=config.TRAINED_MODEL_DIR)

# TRAIN
total_timesteps = 500
stockStrategy.train(env_train, total_timesteps, use_existing=True)

# TEST
e_trade_gym = StockTradingEnv(df=trade_df, turbulence_threshold=70, risk_indicator_col='vix', **env_kwargs)
stockStrategy.test(e_trade_gym)
