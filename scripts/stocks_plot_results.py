from config import general as config

# matplotlib.use('Agg')
# %matplotlib inline

from lib.stocks import StocksStrategy

FILE_PREFIX = "stocks"
ROOT_DIR = '..'
model_name = "SAC"

stockStrategy = StocksStrategy(model_name="SAC", root_dir=ROOT_DIR,
                               results_dir=config.RESULTS_DIR,
                               trained_model_dir=config.TRAINED_MODEL_DIR)

stockStrategy.generate_backtest_results()
stockStrategy.generate_baseline_stats()
stockStrategy.backtest_plot()
