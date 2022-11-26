import os

from finrl.meta.data_processor import DataProcessor
from finrl.meta.env_custom.env_custom import CustomTradingEnv

from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3

MODELS = {"A2C": A2C, "DDPG": DDPG, "TD3": TD3, "SAC": SAC, "PPO": PPO}


def load_dataset(filename, indicators, use_turbulence=False, use_vix=False, time_interval='1d'):
    dp = DataProcessor("file", filename=filename)
    df = dp.download_data([], '', '', time_interval)
    df = dp.clean_data(df)
    if len(indicators) > 0:
        df = dp.add_technical_indicator(df, indicators)
    if use_turbulence:
        df = dp.add_turbulence(df)
    if use_vix:
        df = dp.add_vix(df)
    return df


def data_split(df, start, end, target_date_col="date"):
    """
    split the dataset into training or testing using date
    :param target_date_col: target date column
    :param end: end date (exclusive)
    :param start: start date (inclusive)
    :param df: (df) pandas dataframe
    :return: (df) pandas dataframe
    """
    data = df[(df[target_date_col] >= start) & (df[target_date_col] < end)]
    data = data.sort_values([target_date_col, "tic"], ignore_index=True)
    data.index = data[target_date_col].factorize()[0]
    return data


def generate_yahoo_dataset(name, ticker_list, start_date, end_date, folder='datasets/stocks'):
    print(f"Generating {name} dataset")
    print(f"Loading {len(ticker_list)} stocks")
    print(f"Start: {start_date} End: {end_date}")

    dp = DataProcessor("yahoofinance")
    df = dp.download_data(ticker_list, start_date, end_date, '1d')
    print(df.shape)
    filename = f"{folder}/{name}.csv"
    df = data_split(df, start_date, end_date)
    df.to_csv(filename)
    print(f"File {filename} written.")


def get_train_env(df, env_kwargs) -> CustomTradingEnv:
    kwargs = env_kwargs.copy()
    e_train_gym = CustomTradingEnv(df=df, **kwargs)
    env, _ = e_train_gym.get_sb_env()
    return env


def get_test_env(df, env_kwargs, turb_thres=None) -> CustomTradingEnv:
    kwargs = env_kwargs.copy()
    kwargs['mode'] = 'test'
    return CustomTradingEnv(df=df, turbulence_threshold=turb_thres, **kwargs)


def load_model_from_file(model_name, filename, tensorboard_log):
    model_file_exists = os.path.isfile(f"{filename}.zip")
    if not model_file_exists:
        raise ValueError("NoModelFileAvailableError")

    model_type = MODELS[model_name]
    loaded_model = model_type.load(f"{filename}.zip", tensorboard_log=tensorboard_log)
    print(f"loaded model from {filename}")
    return loaded_model
