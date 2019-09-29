#!/usr/bin/env python
# Custom module for dealing with global project paths and functions related to injesting and accessing raw data

import sys
import os
import ast
import parse_config
import numpy as np
import pandas as pd
from tqdm import tqdm
import pydicom
# import PIL
# from PIL import Image
# import gdcm
# import cv2


TRAIN_DATA_PATH = parse_config.TRAIN_DATA_PATH
TEST_DATA_PATH = parse_config.TEST_DATA_PATH
CSV_PATHS = parse_config.CSV_PATHS


def translate_dicom(filename,path=TRAIN_DATA_PATH,apply_window=None,test=False):
    """
    Transform a medical DICOM file to a standardized pixel based array
    Arguments:
        filename {[type]} -- [description]
        path {string} -- file path to data, set in config.ini
        apply_window {[type]} -- [description]
        test {bool} -- when true read from the test dicom directory
    """
    if test == True:
        path = TEST_DATA_PATH

    if apply_window != None:
        #TODO: take in here a tuple or dict for window/leveling and apply to dcmread function
        apply_window = None

    img = np.array(pydicom.dcmread(path + filename).pixel_array, dtype=float).T
    standardized_array = np.divide(np.subtract(img,img.mean()),img.std())
    return standardized_array


def create_partitions(paths=CSV_PATHS):
    '''
    Builds a dictionary of train/val/test keys and mapped DICOM IDs, returns the obj
    '''
    partition = dict()
    for csv_file in tqdm(paths):
        data = pd.read_csv(os.path.join(CURRENT_DIR, csv_file), header=0)
        value = list(data.filename)
        key = csv_file[:-4]
        partition[key] = value
    
    return partition

def create_labels(paths=CSV_PATHS):
    '''
    Builds three dictionaries of DICOM ID keys and a vectorized list of 6 labels mapped to hemorrhage subtype
    returns a dict object where the keys are the label type (training, etc.) and the value is that label dict
    '''
    labels = dict()
    for csv_file in tqdm(paths):
        data = pd.read_csv(os.path.join(CURRENT_DIR, csv_file), header=0)
        values_dict = dict(zip(data.filename, data.targets))
        key = csv_file[:-4]
        labels[key] = values_dict
    
    return labels
