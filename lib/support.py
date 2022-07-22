import os
from config import general as config


def check_directory_structure(root_folder='.'):
    if not os.path.exists(f"{root_folder}/{config.DATA_SAVE_DIR}"):
        os.makedirs(f"{root_folder}/{config.DATA_SAVE_DIR}")
    if not os.path.exists(f"{root_folder}/{config.TRAINED_MODEL_DIR}"):
        os.makedirs(f"{root_folder}/{config.TRAINED_MODEL_DIR}")
    if not os.path.exists(f"{root_folder}/{config.TENSORBOARD_LOG_DIR}"):
        os.makedirs(f"{root_folder}/{config.TENSORBOARD_LOG_DIR}")
    if not os.path.exists(f"{root_folder}/{config.RESULTS_DIR}"):
        os.makedirs(f"{root_folder}/{config.RESULTS_DIR}")
