#!/usr/bin/env python
# sys.argv[1] is a model name this will also be assigned to the submission csv name
# e.g. baseline-model w/o the file extension, this script assumes a .pb protobuf file extension

import numpy as np
import csv
import sys
import os
from tqdm import tqdm
import pydicom
import tensorflow as tf
import data_flow
from data_loader import DataGenerator
import parse_config

if parse_config.USING_RTX_20XX:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))


MODEL_NAME = os.path.join('../models/', sys.argv[1] + '.pb')
SUBMISSION_NAME = os.path.join('../submissions/', sys.argv[1]+'_predictions.csv')
DATA_DIRECTORY = data_flow.TEST_DATA_PATH
TEST_CSV = "../submissions/phase1_test_filenames.csv"

# Load the saved model and test data
MODEL = tf.keras.models.load_model(MODEL_NAME)
TEST_DATA_GEN = DataGenerator(csv_filename=TEST_CSV, data_path=DATA_DIRECTORY, shuffle=False, batch_size=1, prediction=True)
CUSTOM_OBJECTS = {}
INTRACRANIAL_HEMORRHAGE_SUBTYPES = [
                                    "epidural",
                                    "intraparenchymal",
                                    "intraventricular",
                                    "subarachnoid",
                                    "subdural",
                                    "any"
                                    ]

def normalize_img(img):
    img = (img - np.mean(img)) / np.std(img)
    return img
    
def window_img(img, min=-50, max=100):
    return normalize_img(np.clip(img, min, max))

def transform_dicom(filename):
    # X = np.empty((1, 512, 512, 2))
    ds = pydicom.dcmread(filename)
    
    norm_img = ds.pixel_array.astype(np.float)
    norm_img = normalize_img(np.array(norm_img, dtype=float)) #X[0,:,:,0]
    norm_img = norm_img[:,:,np.newaxis]

    window_center, window_width, intercept, slope = data_flow.get_windowing(ds)
    windowed_img = data_flow.window_image(ds.pixel_array, window_center, window_width, intercept, slope)
    windowed_img = window_img(windowed_img, -100, 100)
    windowed_img = windowed_img[:,:,np.newaxis]

    X = np.concatenate((norm_img, windowed_img), axis=2)
    X = X[np.newaxis,:,:,:]

    return X



def main():
    with open(SUBMISSION_NAME, 'a+', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Id','Label'])

        for filename in tqdm(os.listdir(DATA_DIRECTORY)):
            img = transform_dicom(DATA_DIRECTORY + filename)
            array = MODEL.predict(img)
            prediction = np.squeeze(array)
            print(prediction)

            for subtype in zip(INTRACRANIAL_HEMORRHAGE_SUBTYPES, prediction):
                readable_id = filename[:-4] + "_" + subtype[0]
                # print(readable_id, subtype[1])
                writer.writerow([readable_id, subtype[1]])

if __name__ == "__main__":
    main()
        