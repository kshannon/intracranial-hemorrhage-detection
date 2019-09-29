#!/usr/bin/env python
# Custom module for dealing with global project paths and functions related to injesting and accessing raw data

import sys
import os
from configparser import ConfigParser


# Derive the absolute path from file
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))

config = ConfigParser()
config.read('./config/config.ini')
# stage 1 data
try:
    TRAIN_DATA_PATH = config.get('path','s1_train_path')
    TEST_DATA_PATH = config.get('path','s1_test_path')
    if os.path.isdir(TRAIN_DATA_PATH) == False:
        raise FileNotFoundError("Train path failed, dir not found")
    if os.path.isdir(TEST_DATA_PATH) == False:
        raise FileNotFoundError("Test path failed, dir not found")
except (FileNotFoundError):
    print("Local paths do not exist, trying docker paths...")
    TRAIN_DATA_PATH = config.get('path','docker_train')
    TEST_DATA_PATH = config.get('path','docker_test')

TRAIN_CSV = config.get('path','train_csv_path')
VALIDATE_CSV = config.get('path','validate_csv_path')
TEST_CSV = config.get('path','test_csv_path')
CSV_PATHS = [TRAIN_CSV,VALIDATE_CSV,TEST_CSV]

DOCKER_MODE = config.get('mode','use_docker')
USING_RTX_20XX = config.get('mode','gpu_rtx_20xx')

