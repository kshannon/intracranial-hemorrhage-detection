#!/usr/bin/env python
# Custom module for dealing with global project paths and functions related to injesting and accessing raw data

import sys
import os
import numpy as np
import pandas as pd
import configparser
import pydicom
import PIL
from PIL import Image
import gdcm
import cv2

config = configparser.ConfigParser()
config.read('./config.ini')

# stage 1 data
train_data_path = config['path']['s1_train_path']
test_data_path = config['path']['s1_test_path']
df_path = config['path']['df_path']


def translate_dicom(filename,path=train_data_path,apply_window=None,test=False):
    """
    Transform a medical DICOM file to a standardized pixel based array
    Arguments:
        filename {[type]} -- [description]
        path {string} -- file path to data, set in config.ini
        apply_window {[type]} -- [description]
        test {bool} -- when true read from the test dicom directory
    """
    if test == True:
        path = test_data_path

    if apply_window != None:
        #TODO: take in here a tuple or dict for window/leveling and apply to dcmread function
        apply_window = None

    img = np.array(pydicom.dcmread(path + filename).pixel_array, dtype=float).T
    standardized_array = np.divide(np.subtract(img,img.mean()),img.std())
    return standardized_array


def load_df(df_path):
    """[summary]
    
    Arguments:
        df_path {[type]} -- [description]
    """
    pass