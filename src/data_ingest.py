#!/usr/bin/env python
# Custom module for dealing with global project paths and functions related to injesting and accessing raw data

import sys
import os
import numpy as np
import pydicom
import configparser

config = configparser.ConfigParser()
config.read('./config.ini')

# stage 1 data
train_data_path = config['path']['s1_train_path']
test_data_path = config['path']['s1_test_path']

#TODO: add a window/level option in here
def translate_dicom(filename, path=train_data_path, test=False):
    """
    Transform a medical DICOM file to a standardized pixel based array
    """
    if test == True:
        path = test_data_path

    img = np.array(pydicom.dcmread(path + filename).pixel_array, dtype=float).T
    mean = img.mean()
    std = img.std()
    standardized_array = np.divide(np.subtract(img,mean),std)
    return standardized_array