#!/usr/bin/env python
# Custom module for dealing with global project paths and functions related to injesting and accessing raw data

import sys
import os
import numpy as np
import pydicom

# stage 1 data
s1_test_path = "../data/stage_1_test_images/"
s1_train_path = "../data/stage_1_train_images/"
# stage 2 data
# s2_test_path = "../data/stage_2_test_images/"
# s2_train_path = "../data/stage_2_train_images/"

def translate_dicom(filename):
    """
    Transform a medical DICOM file to a standardized pixel based array
    """
    img = np.array(pydicom.dcmread(test_data_path + filename).pixel_array, dtype=float).T
    mean = img.mean()
    std = img.std()
    standardized_array = np.divide(np.subtract(img,mean),std)
    return standardized_array