#!/usr/bin/env python
# coding: utf-8

import sys
import os
import csv
import numpy as np
import pandas as pd
import pydicom
from multiprocessing import  Pool
from tqdm.auto import tqdm
tqdm.pandas()


TRAIN_CSV_PATH = '../src/training.csv'
VALIDATE_CSV_PATH = '../src/validation.csv'
# TEST_CSV_PATH = '../src/testing.csv'

train_data = '../../data/stage_1_train_images/'
# test_data = '../../data/stage_1_test_images/'


def check_dicom(row, path=train_data):
    try:
        data = pydicom.dcmread(path+row[0])
    except ValueError:
        print('corruption on open...')
        return True
    try:
        img = np.array(data.pixel_array, dtype=float)
    except ValueError:
        print('corruption on pixel_array...')
        return True
    if img.shape != (512, 512):
        print('square peg in round hole!')
        return True
    if np.std(img) == 0:
        print('Zero std dev.')
        return True

    return False


def parallelize_dataframe(df, func, n_cores=4):
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def find_bad_actors(df):
    df['bad_actors'] = df.progress_apply(lambda x: check_dicom(x), axis=1)
    return df


############### Comment or Uncomment to process VALIDATION data ###############
df_validate = pd.read_csv(VALIDATE_CSV_PATH, header=None)
num_rows = df_validate.shape[0]

# find bad actors by reading dicom data
df_validate = parallelize_dataframe(df_validate, find_bad_actors)
df_validate_cleaned = df_validate.loc[df_validate['bad_actors'] == False]

# remove bad actors from the df and check shapes
df_validate_cleaned = df_validate_cleaned.drop(columns=['bad_actors'])
assert df_validate_cleaned.shape[0] <= df_validate.shape[0]

print("Rows in validate before: ", num_rows)
print("Rows in validate after: ", df_validate_cleaned.shape[0])
print("Verified bad actors were removed")

df_validate_cleaned.columns = ["filename", "targets", "any"]
df_validate_cleaned.to_csv('../src/validation_cleaned.csv', index=False)



############### Comment or Uncomment to process TRAINING data ###############
df_train = pd.read_csv(TRAIN_CSV_PATH, header=None)
train_rows = df_train.shape[0]

# find bad actors by reading dicom data
df_train = parallelize_dataframe(df_train, find_bad_actors)
df_train_cleaned = df_train.loc[df_train['bad_actors'] == False]

# remove bad actors from the df and check shapes
df_train_cleaned = df_train_cleaned.drop(columns=['bad_actors'])
assert df_train_cleaned.shape[0] <= df_train.shape[0]

print("Rows in train before: ", train_rows)
print("Rows in train after: ", df_train_cleaned.shape[0])
print("Verified bad actors were removed")

df_train_cleaned.columns = ["filename", "targets", "any"]
df_train_cleaned.to_csv('../src/training_cleaned.csv', index=False)



############### Comment or Uncomment to process TEST data ###############
# df_validate = pd.read_csv(VALIDATE_CSV_PATH, header=None)
# print("Rows in train before: ", df_validate.shape[0])
# df_validate = parallelize_dataframe(df_validate, find_bad_actors)
# df_validate_cleaned = df_validate.loc[df_validate['bad_actors'] == False]
# df_validate_cleaned = df_validate_cleaned.drop(columns=['bad_actors'])
# assert df_validate_cleaned.shape[0] <= df_validate.shape[0]
# print("Rows in train after: ", df_validate_cleaned.shape[0])
# os.remove(VALIDATE_CSV_PATH)
# print("Verified bad actors were removed and deleted old train CSV")
# df_validate_cleaned.to_csv('../src/training.csv', index=False)

print('All Done!')
