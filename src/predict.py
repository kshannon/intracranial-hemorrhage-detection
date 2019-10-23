#!/usr/bin/env python
# sys.argv[1] is the file name that the submission csv will be saved as
# e.g. baseline-model w/o the file extension, this script assumes a .pb protobuf file extension

import numpy as np
import csv
import sys
import os
import math
from tqdm import tqdm
import pydicom
import tensorflow as tf
import data_flow
from data_loader import DataGenerator
import parse_config
# from custom_loss import weighted_log_loss, weighted_loss


# comment out if using tensorflow 2.x
# if parse_config.USING_RTX_20XX:
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     tf.keras.backend.set_session(tf.Session(config=config))

MODEL_PATH = '../models/'
MODEL_EXT = '.pb'
SUBMISSION_NAME = os.path.join('../submissions/', sys.argv[1]+'-predictions.csv')
DATA_DIRECTORY = data_flow.TEST_DATA_PATH
TEST_CSV = "../submissions/phase1_test_filenames.csv"

# Model names are entered here as a list of strings 
# e.g. mobilenetv2-dim999x999-bce-oct1v9
MODELS = {}
MODELS['epidural'] = []
MODELS['intraparenchymal'] = []
MODELS['intraventricular'] = []
MODELS['subarachnoid'] = []
MODELS['any'] = ['mobilenetv2-dim224x224-bce-any-oct22v1']
PREDICT_ONLY_ANY = True #set this to true if you are only loading a model for 'any' subtype


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
        models.append(tf.keras.models.load_model(os.path.join(MODEL_PATH, model + MODEL_EXT)))
        print('Great! Just Loaded: ' + model)
    return models


def predict_on_img(model_array, img, predicting_any=False):
    preds = []
    for model in model_array:
        prediction = model.predict(img)
        preds.append(np.squeeze(prediction))
        prediction = sum(preds) / float(len(preds))
        # force predictions onj subtype 'any' to be 1 or 0
        if predicting_any and prediction > 0.5:
            prediction = math.ceil(prediction)
        if predicting_any and prediction < 0.5:
            prediction = math.floor(prediction)
    return prediction


def main():
    # load models
    any_models = load_models('any')
    if not PREDICT_ONLY_ANY:
        epidural_models = load_models("epidural")
        intraparenchymal_models = load_models("intraparenchymal")
        intraventricular_models = load_models('intraventricular')
        subarachnoid_models = load_models('subarachnoid')
        subdural_models = load_models('subdural')

    with open(SUBMISSION_NAME, 'a+', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Id','Label'])

        for idx in tqdm(TEST_DATA_GEN.indexes):
            img, labels = TEST_DATA_GEN.__getitem__(idx) #ignore labels, just an empty obj
            filename = TEST_DATA_GEN.df.iloc[idx,0]


            # test for subtype any
            pred_any = predict_on_img(any_models, img, predicting_any=True)

            # Handle the case where we predict no occurrence of 'any' IH
            if pred_any == 0 or PREDICT_ONLY_ANY == True:
                filename_pred_vector = [0.0, 0.0, 0.0, 0.0, 0.0, pred_any]
                for filename_pred in zip(INTRACRANIAL_HEMORRHAGE_SUBTYPES, filename_pred_vector):
                    dicom_id = filename[:-4] + "_" + filename_pred[0]
                    # print(readable_id, subtype[1])
                    writer.writerow([dicom_id, filename_pred[1]])
                continue
            
            # Handle the case for when we think there is occurrence of 'any' IH, lets try to determine the type
            if pred_any > 0:
                pred_epidural = predict_on_img(epidural_models, img)
                pred_intraparenchymal = predict_on_img(intraparenchymal_models, img)
                pred_intraventricular = predict_on_img(intraventricular_models, img)
                pred_subarachnoid = predict_on_img(subarachnoid_models, img)
                pred_subdural = predict_on_img(subdural_models, img)
                filename_pred_vector = [pred_epidural,
                                        pred_intraparenchymal,
                                        pred_intraventricular,
                                        pred_subarachnoid,
                                        pred_subdural,
                                        pred_any]
                for filename_pred in zip(INTRACRANIAL_HEMORRHAGE_SUBTYPES, filename_pred_vector):
                    dicom_id = filename[:-4] + "_" + filename_pred[0]
                    # print(readable_id, subtype[1])
                    writer.writerow([dicom_id, filename_pred[1]])
                continue


if __name__ == "__main__":
    main()
        