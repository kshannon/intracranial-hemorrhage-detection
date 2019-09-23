#!/usr/bin/env python
# sys.argv[1] is a model name this will also be assigned to the submission csv name

import numpy as numpy
import csv
import sys
import os
import data_ingest
from tqdm import tqdm
from keras.models import load_model
from model import dice_coef, soft_dice_coef, unet_model, dice_coef_loss

intracranial_hemorrhage_subtypes = [
                                    "epidural",
                                    "intraparenchymal",
                                    "intraventricular",
                                    "subarachnoid",
                                    "subdural",
                                    "any"
                                    ]
custom_objects = {"dice_coef":dice_coef,"dice_coef_loss":dice_coef_loss,"soft_dice_coef":soft_dice_coef}
model_name = '../models/' + sys.argv[1]
submission_name = '../submissions/' + sys.argv[1] + '.csv'


def main():
    # model = load_model(model_name, custom_objects=custom_objects)
    with open(submission_name, 'a+', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(['Id','Label'])

        for filename in tqdm(os.listdir(data_ingest.s1_test_path)): 
            standardized_array = data_ingest.translate_dicom(filename)
            
            #TODO: make predictions for each subtype and all
            prediction = 0.5

            for subtype in intracranial_hemorrhage_subtypes:
                readable_id = filename[:-4] + "_" + subtype
                writer.writerow([readable_id, prediction])

if __name__ == "__main__":
    main()
        


# expanded_array = standardized_array[np.newaxis, ..., np.newaxis]
# msk = model.predict(expanded_array)
# msk = np.squeeze(np.round(msk)) #remove axis w/ dims of 1, round mask for given probbabilties to [0,1]