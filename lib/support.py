import os
from config import general as config
from datetime import datetime


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


def get_run_timestamp():
    return datetime.now().strftime("%m%d%H%M")
