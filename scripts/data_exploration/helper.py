import pandas as pd
import numpy as np

def get_close_list(df):
    return df.pivot(index="date", columns="tic", values="close")

def convert_prices(prices, use_log=False, replace_missing=False):
    X = prices / prices.shift(1).fillna(method="ffill")
    for name, s in X.items():
        X[name].iloc[s.index.get_loc(s.first_valid_index()) - 1] = 1.0

    if replace_missing:
        X = X.fillna(1.0)

    return np.log(X) if use_log else X

def get_data(res='1h'):
    df = pd.read_csv(f'../../datasets/thesis/{res}/all_{res}.csv', index_col=0)
    df['date'] = pd.to_datetime(df['date'])
    return df

def get_parsed_data(res='1h', ds_name='plus'):
    df = pd.read_csv(f'../../datasets/thesis/crypto_{res}_{ds_name}.csv', index_col=0)
    df['date'] = pd.to_datetime(df['date'])
    return df

def get_ticker_name_list(df):
    return df['tic'].unique()