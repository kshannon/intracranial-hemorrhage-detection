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


TRAIN_DATA_PATH = parse_config.TRAIN_DATA_PATH
TEST_DATA_PATH = parse_config.TEST_DATA_PATH
CSV_PATHS = parse_config.CSV_PATHS


def translate_dicom(filename, apply_window=True):
    """
    Transform a medical DICOM file to a standardized pixel based array
    Arguments:
        filename {string}
        path {string} -- file path to data, set in config.ini
        apply_window {bool} -- if True (default) then windowed png of dicom data is returned
    """
    data = pydicom.dcmread(filename)
    
    if apply_window:
        window_center, window_width, intercept, slope = get_windowing(data)
        img = window_image(data.pixel_array, window_center, window_width, intercept, slope)
        return np.array(img, dtype=float)

    img = np.array(data.pixel_array, dtype=float)
    standardized_array = np.divide(np.subtract(img,img.mean()),img.std())
    return standardized_array


def window_image(img, window_center, window_width, intercept, slope):
    """
    Given a CT scan img apply a windowing to the image
    Arguments:
        img {np.array} -- array of a dicom img processed by pydicom.dcmread()
        window_center,window_width,intercept,slope {floats} -- values provided by dicom file metadata
    Attribution: This code comes from Richard McKinley's Kaggle kernel
    """
    img = (img * slope + intercept)
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max
    return img 


def get_first_of_dicom_field_as_int(x):
    """
    Converts pydicom obj into an int
    Arguments:
        x {pydicom obj} -- either a single or multivalue obj
    Attribution: This code comes from Richard McKinley's Kaggle kernel
    """
    #get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)


def get_windowing(data):
    """
    Retrieves windowing data from dicom metadata
    Arguments:
        data {pydicom data obj} -- object returned from pydicom dcmread() 
    Attribution: This code comes from Richard McKinley's Kaggle kernel
    """
    dicom_fields = [data.WindowCenter,
                    data.WindowWidth,
                    data.RescaleIntercept,
                    data.RescaleSlope]
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]
