import os
import time

from stable_baselines3 import A2C
from stable_baselines3 import DDPG
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3 import TD3

import config.crypto as crypto
from finrl.agents.stablebaselines3.drl_agent import DRLAgent
from finrl.meta.data_processor import DataProcessor
from finrl.meta.env_custom.env_custom import CustomTradingEnv
from lib.support import log_duration

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
    kwargs['mode'] = 'train'
    e_train_gym = CustomTradingEnv(df=df, **kwargs)
    env, _ = e_train_gym.get_sb_env()
    return env


def get_test_env(df, env_kwargs, turb_thres=None) -> CustomTradingEnv:
    kwargs = env_kwargs.copy()
    kwargs['mode'] = 'test'
    return CustomTradingEnv(df=df, turbulence_threshold=turb_thres, **kwargs)


def load_model_from_file(model_name, filename, tensorboard_log, device='cpu'):
    model_file_exists = os.path.isfile(f"{filename}.zip")
    if not model_file_exists:
        raise ValueError(f"NoModelFileAvailableError for {filename}")

    model_type = MODELS[model_name]
    loaded_model = model_type.load(f"{filename}.zip", tensorboard_log=tensorboard_log, device=device)
    print(f"loaded model from {filename}")
    return loaded_model


def get_model_params(model_name, run_config):
    if model_name in crypto.MODEL_PARAMS:
        if run_config in crypto.MODEL_PARAMS[model_name]:
            return crypto.MODEL_PARAMS[model_name][run_config]
        print(f"settings for config {run_config} not found - returning default settings for {model_name}")
        return crypto.MODEL_PARAMS[model_name]['default']
    raise ValueError(f"settings for model {model_name} not found")


def train(df, env_kwargs, settings, do_eval=False, df_test=None):
    env = get_train_env(df, env_kwargs)
    agent = DRLAgent(env=env)
    eval_env = None
    if do_eval:
        eval_env = get_test_env(df_test, env_kwargs)

    if settings['retrain_existing_model']:
        print(f"Loading existing model from {settings['previous_model_name']}")
        model = load_model_from_file(env_kwargs['model_name'],
                                     settings['previous_model_name'],
                                     settings['tensorboard_log'],
                                     settings['model_params']['device'])
        model.set_env(env)
    else:
        # initialize new model
        model = agent.get_model(env_kwargs['model_name'],
                                model_kwargs=settings['model_params'],
                                tensorboard_log=settings['tensorboard_log'])

    start = time.time()
    trained_model = agent.train_model(model=model,
                                      eval_env=eval_env,
                                      eval_during_train=do_eval,
                                      tb_log_name=f"{env_kwargs['model_name']}_{env_kwargs['run_name']}",
                                      total_timesteps=settings['total_timesteps'])
    log_duration(start)

    if settings['save_model']:
        print(f"Storing model in {settings['target_model_filename']}")
        trained_model.save(settings['target_model_filename'])

    return trained_model


def test(df, env_kwargs, settings):
    env = get_test_env(df, env_kwargs)
    model = load_model_from_file(env_kwargs['model_name'],
                                 settings['target_model_filename'],
                                 settings['tensorboard_log'],
                                 settings['model_params']['device'])

    start = time.time()
    df_state, df_actions = DRLAgent.DRL_prediction(model=model, environment=env)
    log_duration(start)

    df_state.to_csv(f"{settings['file_prefix']}_state.csv")
    df_actions.to_csv(f"{settings['file_prefix']}_actions.csv")
