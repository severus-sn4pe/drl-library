import time

from config import tickers, stocks
from finrl import config
from lib.drl import generate_yahoo_dataset
from lib.support import check_directory_structure

ROOT_FOLDER = '..'
SINGLE_STOCKS = ["AAPL", "AMZN", "GOOG", "MSFT", "TSLA"]
check_directory_structure(ROOT_FOLDER)
start_time = time.time()

generate_yahoo_dataset("DOW30", tickers.DOW_30_TICKER, stocks.TOTAL_START_DATE, stocks.TOTAL_END_DATE,
                       f"{ROOT_FOLDER}/{config.DATA_SAVE_DIR}/stocks")

print("-------------------------------------------")

generate_yahoo_dataset("NAS15", tickers.NASDAQ_SELECTION, stocks.TOTAL_START_DATE, stocks.TOTAL_END_DATE,
                       f"{ROOT_FOLDER}/{config.DATA_SAVE_DIR}/stocks")

print("-------------------------------------------")

for symbol in SINGLE_STOCKS:
    generate_yahoo_dataset(symbol, [symbol], stocks.TOTAL_START_DATE, stocks.TOTAL_END_DATE,
                           f"{ROOT_FOLDER}/{config.DATA_SAVE_DIR}/stocks")

    print("-------------------------------------------")

print("-------------------------------------------")
print(f"Done in {time.time() - start_time:.2f}s")
print("===========================================")
