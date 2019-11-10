import os
import sys
import argparse
import tensorflow as tf
from tensorflow import keras
import pandas as pd
from data_loader import read_trainset, DataGenerator
import parse_config


# comment out if using tensorflow 2.x
if parse_config.USING_RTX_20XX:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))

parser = argparse.ArgumentParser(description='Intracranial Hemorrhage Stage 2 Inference Script',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_filename', type=str)
args = parser.parse_args()
MODEL_NAME = '../models/' + args.model_filename
img_size = (256,256,3)
batch_size=16 #must be evenly divisiable by number of images in data gen.

# Define paths
test_images_dir = '../../data/stage_2_test_images/'
testset_filename = "../submissions/stage_2_sample_submission.csv"

def read_testset(filename):
    """ Read the submission sample csv
        Args:
            filename (str): Filename of the sample submission 
        Returns:
            df (panda dataframe):  Return a dataframe for inference.  

     """
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)

    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)

    return df

def create_submission(model, data, test_df):

    print('+'*50)
    print("Creating predictions on test dataset")
    pred = model.predict_generator(data, verbose=1)
    out_df = pd.DataFrame(pred, index=test_df.index, columns=test_df.columns)
    test_df = out_df.stack().reset_index()
    test_df.insert(loc=0, column='ID', value=test_df['Image'].astype(str) + "_" + test_df['Diagnosis'])
    test_df = test_df.drop(["Image", "Diagnosis"], axis=1)
    print("Saving submissions to submission.csv")
    test_df.to_csv('../submissions/stage2-final-submission-v2.csv', index=False)

    return test_df

def main():
    test_df = read_testset(testset_filename)
    test_generator = DataGenerator(list_IDs = test_df.index, 
                                    batch_size = batch_size,
                                    img_size = img_size,
                                    img_dir = test_images_dir)
    best_model = keras.models.load_model(MODEL_NAME, compile=False)
    create_submission(best_model, test_generator, test_df)


if __name__ == "__main__":
    main()