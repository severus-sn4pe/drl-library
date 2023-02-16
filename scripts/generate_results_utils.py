import os
from datetime import timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

crypto_names = ['AVAXUSDT', 'AXSUSDT', 'BTCUSDT', 'DOGEUSDT', 'ETHUSDT', 'LINKUSDT', 'LTCUSDT', 'SHIBUSDT', 'TLMUSDT', 'UNIUSDT', 'cash']

weight_colors = ["#FF0000", "#18E3FF", "#FFA90C", "#D9E501", "#0A2DC2", "#05A41B", "#878787", "#935948", "#b372fa", "#FC12DD", "#CCCCCC"]

def get_immediate_subdirectories(a_dir):
    return [name for name in os.listdir(a_dir)
            if os.path.isdir(os.path.join(a_dir, name))]


def get_delta(negative=False, res='1d'):
    hours = 24
    if res == '6h':
        hours = 6
    if res == '12h':
        hours = 12
    if res == '1h':
        hours = 1
    if negative:
        hours *= -1
    return timedelta(hours=hours)


def get_portfolio_value(portfolio_prefix):
    value_suffix = '_state.csv'
    df = pd.read_csv(f"{portfolio_prefix}{value_suffix}")  # , index_col=0
    df = df.rename(columns={df.columns[0]: 'date'})
    df.loc[0, 'date'] = pd.to_datetime(df.loc[1]['date']) - timedelta(days=1)
    df['date'] = pd.to_datetime(df['date'])
    df['return'] = df['account_value'].pct_change() + 1
    df['equity'] = (df['return']).cumprod()
    df.loc[0, 'return'] = 1
    df.loc[0, 'equity'] = 1
    df.index = df['date']
    df = df.drop('date', axis=1)
    return df


def get_actions(portfolio_prefix):
    actions_suffix = '_actions.csv'
    df2 = pd.read_csv(f"{portfolio_prefix}{actions_suffix}", index_col=0)
    df2.index = pd.to_datetime(df2.index)
    return df2


def get_prices(start_date, end_date, res='1d'):
    price_filename = f'../datasets/thesis/{res}/all_{res}.csv'
    df3 = pd.read_csv(price_filename)
    df3['date'] = pd.to_datetime(df3['date'])
    # get or column with close price for each stock per tick
    df3 = df3.pivot(index="date", columns="tic", values="close")
    # limit dataset to portfolio range
    df3 = df3[(df3.index >= start_date) & (df3.index <= end_date)]
    # get relative change from price
    df3 = df3.pct_change()
    # set first row to 0
    df3.loc[df3.iloc[0].name] = 0
    # get cumulative return for each stock
    df3 = (df3 + 1).cumprod()
    return df3


def get_benchmark(start_date, end_date, res='1d'):
    benchmark_returns_1d = '../datasets/thesis/benchmark/1d_benchmark_returns.csv'
    benchmark_returns_1h = '../datasets/thesis/benchmark/1h_benchmark_returns.csv'
    # if res == '1d':
    #     filename = benchmark_returns_1d
    # else:
    #     filename = benchmark_returns_1h
    filename = benchmark_returns_1d
    df_benchmark = pd.read_csv(f"{filename}")
    df_benchmark.index = pd.to_datetime(df_benchmark['date'])
    df_benchmark = df_benchmark.drop('date', axis=1)
    df_benchmark = limit_dataframe(df_benchmark, start_date, end_date)
    return df_benchmark


def limit_dataframe(df, start_date, end_date):
    return df[(df.index >= start_date) & (df.index <= end_date)].copy()


def merge_strategy_and_benchmark_returns(df_value, df_bench):
    df_value['strategy'] = df_value['return']
    df_strategy = df_value[['strategy']]
    # df_benchmark = limit_dataframe(df_bench, df_strategy.index[0], df_strategy.index[-1])
    df_returns = df_strategy.merge(df_bench, how='left', left_index=True, right_index=True)
    return df_returns


def get_cumulative_returns(df):
    return df.cumprod()


def get_asset_ratios(acc_val, res='1d'):
    names = ['AVAXUSDT', 'AXSUSDT', 'BTCUSDT', 'DOGEUSDT', 'ETHUSDT', 'LINKUSDT', 'LTCUSDT', 'SHIBUSDT', 'TLMUSDT', 'UNIUSDT']
    ratio_df = acc_val[["date", "cash"]].copy()
    ratio_df['date'] = (pd.to_datetime(ratio_df['date']) + get_delta(res=res)).dt.strftime("%Y-%m-%d")  # if res == '1d' else "%Y-%m-%d %X")
    for n in names:
        ratio_df[n] = acc_val[f"{n}_price"] * acc_val[f"{n}_amount"]
        # ratio_df[n] = ratio_df[n].where(ratio_df[n] < 0, 0)
    # ratio_df['all_assets_without_cash'] = ratio_df[[f"{x}" for x in names]].sum(axis=1)
    ratio_df[names] = ratio_df[names].clip(lower=0)
    for n in names:
        ratio_df[n] = ratio_df[n] / acc_val[f"account_value"]
        
    ratio_df["cash"] = ratio_df["cash"] / acc_val[f"account_value"]
    ratio_df["all_assets"] = ratio_df[names].sum(axis=1)
    return ratio_df


def convert_ratios_to_value(ratio, results):
    new_ratio = ratio.copy()
    for name in crypto_names:
        new_ratio[name] = new_ratio[name] * results['account_value']
    return new_ratio


def print_weights(ratio_df, save_location="", title="", show=False, xlabel_split=6):
    fig = plt.figure(figsize=(12,4), tight_layout=True)
    frame1 = plt.gca()
    plt.stackplot(ratio_df.date, *[ratio_df[col] for col in names], labels=(names), colors=weight_colors, baseline='zero')
    plt.legend(ncols=1, bbox_to_anchor=(1.2, 1.0))
    plt.grid(True, which='major')
    
    # reduce amount of xtick labels
    tx = plt.xticks()
    max_items = len(tx[0])
    iteration_step = int(np.floor(max_items/xlabel_split))
    tick_idx_new, tick_labels_new = [], []
    for i in range(xlabel_split):
        tick_idx_new.append(tx[0][i*iteration_step])
        tick_labels_new.append(tx[1][i*iteration_step])
    tick_idx_new.append(tx[0][-1])
    tick_labels_new.append(tx[1][-1])
    
    plt.xticks(tick_idx_new, tick_labels_new, rotation=90)
    plt.margins(x=0, y=0)
    plt.legend(bbox_to_anchor=(1, 1), loc=2, borderaxespad=1, fancybox=False, frameon=False)
    if ratio_df[names[-1]].max() > 1:
        plt.hlines(1e6, ratio_df.date.iloc[0], ratio_df.date.iloc[-1], color='#000', linewidths=1, linestyles='dashed')
    if title != "":
        plt.title(title)
    
    if save_location != "":
        plt.savefig(save_location)
    
    if show:
        plt.show()
    plt.close()
