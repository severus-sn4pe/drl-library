import os
from config import general as config
from datetime import datetime
import time


def check_directory_structure(root_folder='.'):
    if not os.path.exists(f"{root_folder}/{config.DATA_SAVE_DIR}"):
        os.makedirs(f"{root_folder}/{config.DATA_SAVE_DIR}")
    if not os.path.exists(f"{root_folder}/{config.TRAINED_MODEL_DIR}"):
        os.makedirs(f"{root_folder}/{config.TRAINED_MODEL_DIR}")
    if not os.path.exists(f"{root_folder}/{config.TENSORBOARD_LOG_DIR}"):
        os.makedirs(f"{root_folder}/{config.TENSORBOARD_LOG_DIR}")
    if not os.path.exists(f"{root_folder}/{config.RESULTS_DIR}"):
        os.makedirs(f"{root_folder}/{config.RESULTS_DIR}")


def check_run_directory_structure(root_dir, results_dir='results', strat_name=None, model_name=None, run_name=None):
    if strat_name is not None:
        if not os.path.exists(f"{root_dir}/{results_dir}/{strat_name}"):
            os.mkdir(f"{root_dir}/{results_dir}/{strat_name}")
        if model_name is not None:
            if not os.path.exists(f"{root_dir}/{results_dir}/{strat_name}/{model_name}"):
                os.mkdir(f"{root_dir}/{results_dir}/{strat_name}/{model_name}")
            if run_name is not None:
                if not os.path.exists(f"{root_dir}/{results_dir}/{strat_name}/{model_name}/{run_name}"):
                    os.mkdir(f"{root_dir}/{results_dir}/{strat_name}/{model_name}/{run_name}")

def get_time():
    now = datetime.now()
    return now.strftime("%d.%m.%Y %H:%M:%S")


def get_duration(duration):
    m, s = divmod(duration, 60)
    h, m = divmod(m, 60)
    return f'{h:02.0f}:{m:02.0f}:{s:02.0f}'


def log_duration(start_time):
    print(f"{get_time()}: finished in {get_duration(time.time() - start_time)}")
