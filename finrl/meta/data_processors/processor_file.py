from __future__ import annotations

import itertools

import exchange_calendars as tc
import numpy as np
import pandas as pd
from stockstats import StockDataFrame as Sdf

from finrl.meta.data_processors.processor_yahoofinance import (YahooFinanceProcessor as YahooFinance)


class FileProcessor:
    """
    Provides methods for retrieving stock data from file system
    """

    def __init__(self, filename):
        self.start = '2014-01-01'
        self.end = '2021-12-31'
        self.time_interval = '1d'
        self.filename = filename
        pass

    def download_data(self, start_date: str, end_date: str, ticker_list: list, time_interval: str) -> pd.DataFrame:
        self.start = start_date
        self.end = end_date
        self.time_interval = time_interval
        # print(self.filename)
        df = pd.read_csv(self.filename, index_col=0)
        print(f"Loaded {df.shape[0]} rows from file {self.filename}")
        return df

    def clean_data(self, data) -> pd.DataFrame:
        # data = data.fillna(method="ffill").fillna(method="bfill")
        # data = data.fillna(0)
        df = data.copy()
        df = df.sort_values(["date", "tic"], ignore_index=True)
        df.index = df.date.factorize()[0]
        merged_closes = df.pivot_table(index="date", columns="tic", values="close")
        merged_closes = merged_closes.dropna(axis=1)
        tics = merged_closes.columns
        df = df[df.tic.isin(tics)]

        data = df
        list_ticker = data["tic"].unique().tolist()
        list_date = list(pd.date_range(data['date'].min(), data['date'].max(), freq=self.time_interval).astype(str))
        combination = list(itertools.product(list_date, list_ticker))

        processed_full = pd.DataFrame(combination, columns=["date", "tic"]).merge(data, on=["date", "tic"], how="left")
        processed_full = processed_full[processed_full['date'].isin(data['date'])]
        processed_full = processed_full.sort_values(['date', 'tic'])

        processed_full = processed_full.fillna(0)

        processed_full.sort_values(['date', 'tic'], ignore_index=True)
        return processed_full

    def add_technical_indicator(self, data, tech_indicator_list):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        print(f"Adding {len(tech_indicator_list)} indicators: {', '.join(tech_indicator_list)}")
        df = data.copy()
        df = df.rename(columns={"date": "time"})
        df = df.sort_values(by=["tic", "time"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in tech_indicator_list:
            print(f"adding {indicator}")
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["time"] = df[df.tic == unique_ticker[i]]["time"].to_list()
                    indicator_df = indicator_df.append(temp_indicator, ignore_index=True)
                except Exception as e:
                    print(e)
            df = df.merge(indicator_df[["tic", "time", indicator]], on=["tic", "time"], how="left")
        df = df.sort_values(by=["time", "tic"])
        df = df.rename(columns={"time": "date"})
        df = df.fillna(method='ffill').fillna(method='bfill')
        return df

    def add_turbulence(self, data):
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        print(f"Adding turbulence index")
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df)
        df = df.merge(turbulence_index, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def calculate_turbulence(self, data, time_period=252):
        """calculate turbulence index"""
        # fixed look back period of 60 days, scaled depending on selected time_interval
        if self.time_interval == '1d':
            time_period = 60
        if self.time_interval == '6h':
            time_period = 60 * 4
        if self.time_interval == '12h':
            time_period = 60 * 2
        if self.time_interval == '1h':
            time_period = 60 * 24
        if self.time_interval == '30min':
            time_period = 60 * 24 * 2
        if self.time_interval == '5min':
            time_period = 60 * 24 * 12
        if self.time_interval == '1min':
            time_period = 60 * 24 * 60
        print(f"Calculating Turbulence for {self.time_interval} res with lookback_period={time_period}")
        df = data.copy()
        df_price_pivot = df.pivot(index="date", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.date.unique()
        # start after a year
        start = time_period
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - time_period])
                ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[hist_price.isna().sum().min():].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )
            # cov_temp = hist_price.cov()
            # current_temp=(current_price - np.mean(hist_price,axis=0))

            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(current_temp.values.T)
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame({"date": df_price_pivot.index, "turbulence": turbulence_index})
        return turbulence_index

    def add_vix(self, data):
        print(f"Adding vix")
        df = data.copy()
        yahoo_dp = YahooFinance()
        df_vix = yahoo_dp.download_data(start_date=df.date.min(), end_date=df.date.max(),
                                        ticker_list=["^VIX"], time_interval='1d')
        vix = df_vix[["date", "close"]]
        vix.columns = ["date", "vix"]
        df = df.merge(vix, on="date")
        df = df.sort_values(["date", "tic"]).reset_index(drop=True)
        return df

    def df_to_array(self, df, tech_indicator_list, if_vix):
        """transform final df to numpy arrays"""
        unique_ticker = df.tic.unique()
        print(unique_ticker)
        if_first_time = True
        for tic in unique_ticker:
            if if_first_time:
                price_array = df[df.tic == tic][["adjcp"]].values
                # price_ary = df[df.tic==tic]['close'].values
                tech_array = df[df.tic == tic][tech_indicator_list].values
                if if_vix:
                    turbulence_array = df[df.tic == tic]["vix"].values
                else:
                    turbulence_array = df[df.tic == tic]["turbulence"].values
                if_first_time = False
            else:
                price_array = np.hstack([price_array, df[df.tic == tic][["adjcp"]].values])
                tech_array = np.hstack([tech_array, df[df.tic == tic][tech_indicator_list].values])
        assert price_array.shape[0] == tech_array.shape[0]
        assert tech_array.shape[0] == turbulence_array.shape[0]
        print("Successfully transformed into array")
        return price_array, tech_array, turbulence_array

    def get_trading_days(self, start, end):
        nyse = tc.get_calendar("NYSE")
        df = nyse.sessions_in_range(
            # pd.Timestamp(start, tz=pytz.UTC), pd.Timestamp(end, tz=pytz.UTC)
            pd.Timestamp(start),
            pd.Timestamp(end),
            # bug fix:ValueError: Parameter `start` received with timezone defined as 'UTC' although a Date must be timezone naive.
        )
        trading_days = []
        for day in df:
            trading_days.append(str(day)[:10])

        return trading_days
