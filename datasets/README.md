# datasets

a set of various financial datasets, which are stored outside of git and have to be downloaded manually.

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

## Cryptocurrencies

* files in ``crypto/``:
* datasource: ``binance``
* data resolution: ``1d, 1h, 5min``
* columns: ``index, date (yyyy-mm-dd hh:mm:ss), open, high, low, close, volume, tic, day``

| Dataset      | Start Date          | End Date            |
|--------------|---------------------|---------------------|
| ``ADAUSDT``  | 2018-04-17 04:02:00 | 2022-07-20 23:59:00 |
| ``AVAXUSDT`` | 2020-09-22 06:30:00 | 2022-07-20 23:59:00 |
| ``BNBUSDT``  | 2017-11-06 03:54:00 | 2022-07-20 23:59:00 |
| ``BTCUSDT``  | 2017-08-17 04:00:00 | 2022-07-20 23:59:00 |
| ``DOGEUSDT`` | 2019-07-05 12:00:00 | 2022-07-20 23:59:00 |
| ``DOTUSDT``  | 2020-08-18 23:00:00 | 2022-07-20 23:59:00 |
| ``ETHUSDT``  | 2017-08-17 04:00:00 | 2022-07-20 23:59:00 |
| ``SOLUSDT``  | 2020-08-11 06:00:00 | 2022-07-20 23:59:00 |
| ``UNIUSDT``  | 2020-09-17 03:00:00 | 2022-07-20 23:59:00 |
| ``XRPUSDT``  | 2018-05-04 08:11:00 | 2022-07-20 23:59:00 |




### Datasets of single cryptocurrencies

Available stocks : ``'BTCUSDT', 'ETHUSDT', 'ADAUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT',
'DOTUSDT', 'DOGEUSDT', 'AVAXUSDT', 'UNIUSDT'`` 

see files `crypto/{symbol}_{data_resolution}.csv`