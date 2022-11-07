# datasets

a set of various financial datasets, which are stored outside of git and have to be downloaded manually.

## Cryptocurrencies

* files in ``thesis/``:
* datasource: ``binance``
* data resolution: ``1d, 1h, 30min, 5min, 1min``
* columns: ``index, date (yyyy-mm-dd [hh:mm:ss]), open, high, low, close, volume, tic, day``
* missing data in OHLC are forward-filled with the last known closing price of the asset with 0 volume

data storage urls in ``thesis/all_data_{data_resolution}.txt``

### Crypto-Index for Thesis 

10 Cryptocurrencies in index: ``'BTCUSDT', 'ETHUSDT', 'LTCUSDT', 'AVAXUSDT', 'LINKUSDT', 'UNIUSDT', 'TLMUSDT', 'AXSUSDT', 'DOGEUSDT', 'SHIBUSDT'``

Start: ``2021-05-11`` End: `2022-11-01`

see files `thesis/{data_resolution}/all_{data_resolution}.csv`

### Datasets of single cryptocurrencies

Available single cryptocurrencies: *from list above*

see files `thesis/{data_resolution}/{symbol}_{data_resolution}.csv`


## Stocks

* files in ``stocks/``
* datasource: ``yfinance``
* data resolution: ``1 day``
* columns: ``index, date (yyyy-mm-dd), open, high, low, close, volume, tic, day``

| Dataset | Start Date | End Date   |
|---------|------------|------------|
| `DOW30` | 2009-01-01 | 30.06.2022 |
| `NAS15` | 2009-01-01 | 30-06-2022 |
| `AAPL`  | 2009-01-01 | 30-06-2020 |
| `AMZN`  | 2009-01-01 | 30-06-2022 |
| `GOOG`  | 2014-03-27 | 30-06-2022 |
| `MSFT`  | 2009-01-01 | 30-06-2022 |
| `TSLA`  | 2010-06-29 | 30-06-2022 |

### DOW30

Dow 30 constituents in 2021/10

Included Symbols: ``    "AXP", "AMGN", "AAPL", "BA", "CAT", "CSCO", "CVX", "GS",
"HD", "HON", "IBM", "INTC", "JNJ", "KO", "JPM", "MCD", "MMM", "MRK", "MSFT",
"NKE", "PG", "TRV", "UNH", "CRM", "VZ", "V", "WBA", "WMT", "DIS", "DOW",``

see file ``stocks/DOW30.csv``

### NAS15

Arbitrary selection of 15 tech stocks of NASDAQ

Included Symbols: ``"AAPL", "MSFT", "AMZN", "TSLA", "NVDA", "NFLX", "INTC", "EBAY",
"GOOG", "ABNB", "PYPL", "SBUX", "ATVI", "AMD", "CSCO"``

see file ``stocks/NAS15.csv``

### Datasets of single stocks

Available stocks : ``"AAPL", "AMZN", "GOOG", "MSFT", "TSLA"`` in `stocks/{symbol}.csv`



