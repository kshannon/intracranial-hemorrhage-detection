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
from custom_loss import multilabel_loss 


if parse_config.USING_RTX_20XX:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))

MODEL_NAME = os.path.join('../models/', sys.argv[1] + '.pb')
SUBMISSION_NAME = os.path.join('../submissions/', sys.argv[1]+'-predictions.csv')
DATA_DIRECTORY = data_flow.TEST_DATA_PATH
TEST_CSV = "../submissions/phase1_test_filenames.csv"

# Load the saved model and test data
CUSTOM_OBJECTS = {"multilabel_loss":multilabel_loss}
MODEL = tf.keras.models.load_model(MODEL_NAME, custom_objects=CUSTOM_OBJECTS)
BATCH_SIZE = 1
DIMS = (512,512)
# RESIZE = (224,224) #comment out if not needed and erase param below
TEST_DATA_GEN = DataGenerator(csv_filename=TEST_CSV,
                                data_path=DATA_DIRECTORY,
                                shuffle=False,
                                batch_size=1,
                                prediction=True,
                                resize=None,
                                dims=DIMS)
INTRACRANIAL_HEMORRHAGE_SUBTYPES = ["epidural",
                                    "intraparenchymal",
                                    "intraventricular",
                                    "subarachnoid",
                                    "subdural",
                                    "any"]



def main():
    with open(SUBMISSION_NAME, 'a+', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Id','Label'])

        for idx in tqdm(TEST_DATA_GEN.indexes):
            images, labels = TEST_DATA_GEN.__getitem__(idx)
            prediction = MODEL.predict(images)
            filename = TEST_DATA_GEN.df.iloc[idx,0]

            print(np.squeeze(prediction))
        
            for subtype in zip(INTRACRANIAL_HEMORRHAGE_SUBTYPES, np.squeeze(prediction)):
                readable_id = filename[:-4] + "_" + subtype[0]
                # print(readable_id, subtype[1])
                writer.writerow([readable_id, subtype[1]])



if __name__ == "__main__":
    main()
        