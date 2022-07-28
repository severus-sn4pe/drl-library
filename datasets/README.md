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
* columns: ``index, date (yyyy-mm-dd [hh:mm:ss]), open, high, low, close, volume, tic, day``
* missing data in OHLC are forward-filled with the last known closing price of the asset with 0 volume

| Dataset       | Start Date          | End Date            | IDX10 | IDX20 |
|---------------|---------------------|---------------------|-------|-------|
| ``crypto_1d`` | 2019-07-06 00:00:00 | 2022-06-30 23:59:00 |       |       |
| ``crypto_1h`` | 2019-07-06 00:00:00 | 2022-06-30 23:59:00 |       |       |
|               |                     |                     |       |       |
| ``ADAUSDT``   | 2018-04-18 00:00:00 | 2022-07-20 23:59:00 | X     | X     |
| ``ALGOUSDT``  | 2019-06-22 00:00:00 | 2022-07-20 23:00:00 |       | X     |
| ``APEUSDT``   | 2022-03-18 00:00:00 | 2022-07-20 23:00:00 |       |       |
| ``ATOMUSDT``  | 2019-04-30 00:00:00 | 2022-07-20 23:00:00 |       | X     |
| ``AVAXUSDT``  | 2020-09-23 00:00:00 | 2022-07-20 23:59:00 |       |       |
| ``BNBUSDT``   | 2017-11-07 00:00:00 | 2022-07-20 23:59:00 | X     | X     |
| ``BTCUSDT``   | 2017-08-18 00:00:00 | 2022-07-20 23:59:00 | X     | X     |
| ``DOGEUSDT``  | 2019-07-06 00:00:00 | 2022-07-20 23:59:00 | X     | X     |
| ``DOTUSDT``   | 2020-08-19 00:00:00 | 2022-07-20 23:59:00 |       |       |
| ``EOSUSDT``   | 2018-05-29 00:00:00 | 2022-07-20 23:00:00 |       | X     |
| ``ETCUSDT``   | 2018-06-13 00:00:00 | 2022-07-20 23:00:00 |       | X     |
| ``ETHUSDT``   | 2017-08-18 00:00:00 | 2022-07-20 23:59:00 | X     | X     |
| ``FTMUSDT``   | 2019-06-12 00:00:00 | 2022-07-20 23:00:00 |       | X     |
| ``IOTAUSDT``  | 2018-06-01 00:00:00 | 2022-07-20 23:00:00 |       | X     |
| ``LINKUSDT``  | 2019-01-17 00:00:00 | 2022-07-20 23:00:00 | X     | X     |
| ``LTCUSDT``   | 2017-12-14 00:00:00 | 2022-07-20 23:00:00 | X     | X     |
| ``MANAUSDT``  | 2020-08-07 00:00:00 | 2022-07-20 23:00:00 |       |       |
| ``MATICUSDT`` | 2019-04-27 00:00:00 | 2022-07-20 23:00:00 |       | X     |
| ``NEARUSDT``  | 2020-10-15 00:00:00 | 2022-07-20 23:00:00 |       |       |
| ``NEOUSDT``   | 2017-11-21 00:00:00 | 2022-07-20 23:00:00 | X     | X     |
| ``RUNEUSDT``  | 2020-09-05 00:00:00 | 2022-07-20 23:00:00 |       |       |
| ``SANDUSDT``  | 2020-08-15 00:00:00 | 2022-07-20 23:00:00 |       |       |
| ``SHIBUSDT``  | 2021-05-11 00:00:00 | 2022-07-20 23:00:00 |       |       |
| ``SOLUSDT``   | 2020-08-12 00:00:00 | 2022-07-20 23:59:00 |       |       |
| ``TRXUSDT``   | 2018-06-12 00:00:00 | 2022-07-20 23:00:00 |       | X     |
| ``UNIUSDT``   | 2020-09-18 00:00:00 | 2022-07-20 23:59:00 |       |       |
| ``WAVESUSDT`` | 2019-01-19 00:00:00 | 2022-07-20 23:00:00 |       | X     |
| ``XLMUSDT``   | 2018-06-01 00:00:00 | 2022-07-20 23:00:00 |       | X     |
| ``XMRUSDT``   | 2019-03-16 00:00:00 | 2022-07-20 23:00:00 | X     | X     |
| ``XRPUSDT``   | 2018-05-05 00:00:00 | 2022-07-20 23:59:00 | X     | X     |


### Datasets of single cryptocurrencies

Available cryptocurrencies: *see list above*

see files `crypto/{symbol}_{data_resolution}.csv`