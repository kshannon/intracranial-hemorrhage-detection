#!/usr/bin/env python
# coding: utf-8

import sys
import os
import csv
import numpy as np
import pandas as pd
import pydicom
from tqdm.auto import tqdm
tqdm.pandas()


TRAIN_CSV_PATH = '../src/training.csv'
VALIDATE_CSV_PATH = '../src/validation.csv'
TEST_CSV_PATH = '../src/testing.csv'

train_data = '../../data/stage_1_train_images/'
test_data = '../../data/stage_1_test_images/'


def check_label(row):
    if row[1] == "[0. 0. 0. 0. 0. 0.]":
        return False
    else:
        return True
    

# Comment or Uncomment to process TRAINING data
df_train = pd.read_csv(TRAIN_CSV_PATH, names=['id','label'])
df_train['has_hemorrhage'] = df_train.progress_apply(lambda x: check_label(x), axis=1)
df_train_class_1 = df_train[df_train.has_hemorrhage != False]
df_train_class_1 = df_train_class_1.drop(columns=['has_hemorrhage'])
df_train_class_1.to_csv('../src/class_one_training.csv', index=False, header=False)

# # Comment or Uncomment to process VALIDATION data
#TODO: for validation if required

# Comment or Uncomment to process TEST data
#TODO: for test if required

print('All Done!')