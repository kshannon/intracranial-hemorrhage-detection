#!/usr/bin/env python
# Custom module for dealing with global project paths and functions related to injesting and accessing raw data

import os
import sys
import csv
import pandas as pd
import numpy as np
from tqdm import tqdm


csv_directory = "../submissions/"
DATA_DIRECTORY = "../../data/stage_1_test_images/"

with open(csv_directory + 'phase1_test_filenames.csv', 'w') as outfile:
    writer = csv.writer(outfile)
    for filename in tqdm(os.listdir(DATA_DIRECTORY)):
        writer.writerow([filename])
