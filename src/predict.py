#!/usr/bin/env python
# sys.argv[1] is the file name that the submission csv will be saved as
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
from custom_loss import weighted_log_loss, weighted_loss



if parse_config.USING_RTX_20XX:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))

MODEL_PATH = '../models/'
MODEL_EXT = '.pb'
SUBMISSION_NAME = os.path.join('../submissions/', sys.argv[1]+'-predictions.csv')
DATA_DIRECTORY = data_flow.TEST_DATA_PATH
TEST_CSV = "../submissions/phase1_test_filenames.csv"


# Load the saved model and test data
# CUSTOM_OBJECTS = {"weighted_log_loss":weighted_log_loss, "weighted_loss":weighted_loss}
# MODEL_NAME = os.path.join('../models/', sys.argv[1] + '.pb')
# MODEL = tf.keras.models.load_model(MODEL_NAME, custom_objects=CUSTOM_OBJECTS)



# Model names are entered here as a list of strings 
# e.g. mobilenetv2-dim999x999-bce-oct1v9
MODELS = {}
MODELS['epidural'] = []
MODELS['intraparenchymal'] = []
MODELS['intraventricular'] = []
MODELS['subarachnoid'] = []
MODELS['any'] = ['mobilenetv2-dim244x244-bce-oct22v1']


BATCH_SIZE = 1
DIMS = (224,224)
TEST_DATA_GEN = DataGenerator(csv_filename=TEST_CSV,
                                data_path=DATA_DIRECTORY,
                                shuffle=False,
                                batch_size=1,
                                prediction=True,
                                sigmoid = True,
                                dims=DIMS)
INTRACRANIAL_HEMORRHAGE_SUBTYPES = ["epidural",
                                    "intraparenchymal",
                                    "intraventricular",
                                    "subarachnoid",
                                    "subdural",
                                    "any"]


def load_models(subtype):
    models = []
    for model in MODELS[subtype]:
        loaded_model = tf.keras.models.load_model(os.path.join(MODEL_PATH, model + MODEL_EXT))
        print('Great! Just Loaded: ' + model)
        models.append(loaded_model)
    return models

def predict_ih_subtype(model_array):
    pass
    # do the thing, i.e. TODO: for each model predictict, then abg the preictions then return it
    return prediction


def main():
    # load models
    model_epidural = load_model(subtype)
    model_intraparenchymal = load_model(subtype)
    model_intraventricular = load_model(subtype)
    model_subarachnoid = load_model(subtype)
    model_subdural = load_model(subtype)
    model_any = load_model(subtype)

    with open(SUBMISSION_NAME, 'a+', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Id','Label'])

        for idx in tqdm(TEST_DATA_GEN.indexes):
            images, labels = TEST_DATA_GEN.__getitem__(idx) #ignore labels, just an empty obj
            filename = TEST_DATA_GEN.df.iloc[idx,0]


            # test for any class
            any_prediction = []
            for model in model_any:
                prediction = model.predict(images)
                any.predictions.append(np.squeeze(prediction))
            if len(any_prediction) > 1:
                # TODO: avg_any_prediction = avergae or sigmoid the predictions list for any
            
            if any_prediction = 0:
                for subtype in zip(INTRACRANIAL_HEMORRHAGE_SUBTYPES, [0.0, 0.0, 0.0, 0.0, 0.0, any_prediction]):
                    readable_id = filename[:-4] + "_" + subtype[0]
                    # print(readable_id, subtype[1])
                    writer.writerow([readable_id, subtype[1]])
                continue
            else:

            # TODO:
                # make predictions on other subtype models
                # write rows to csv

                for subtype in zip(INTRACRANIAL_HEMORRHAGE_SUBTYPES, [0.0, 0.0, 0.0, 0.0, 0.0, any_prediction]):
                    readable_id = filename[:-4] + "_" + subtype[0]
                    # print(readable_id, subtype[1])
                    writer.writerow([readable_id, subtype[1]])
                continue



            # print(np.squeeze(prediction))
        
            # for subtype in zip(INTRACRANIAL_HEMORRHAGE_SUBTYPES, np.squeeze(prediction)):
            #     readable_id = filename[:-4] + "_" + subtype[0]
            #     # print(readable_id, subtype[1])
            #     writer.writerow([readable_id, subtype[1]])



if __name__ == "__main__":
    main()
        