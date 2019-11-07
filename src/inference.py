import os, argparse
import tensorflow as tf
from datetime import datetime
from model import MyDeepModel
from data_loader import read_trainset, DataGenerator

from tensorflow import keras
import pandas as pd

parser = argparse.ArgumentParser(description='Intracranial Hemorrhage Stage 2 Inference Script',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--model_filename', type=int, default=40,
                    help='number of epochs to train')
args = parser.parse_args()

# Define paths
test_images_dir = '../../data/stage_2_test_images/'
testset_filename = "../../stage_2_sample_submission.csv"

# MODELS = {
#     'epidural': [],
#     'intraparenchymal': [],
#     'intraventricular': [],
#     'subarachnoid': [],
#     'any': ['mobilenetv2-dim224x224-bce-any-oct22v1']
# }

num_epochs = 10
img_shape = (256,256,3)
batch_size=32

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

# def load_models(subtype):
#     models = []
#     for model in MODELS[subtype]:
#         models.append(tf.keras.models.load_model(os.path.join(MODEL_PATH, model + MODEL_EXT)))
#         print('Great! Just Loaded: ' + model)
#     return models

def create_submission(model, data, test_df):

    print("Creating predictions on test dataset")
    pred = model.predict_generator(data, verbose=1)
    out_df = pd.DataFrame(pred, index=test_df.index, columns=test_df.columns)
    test_df = out_df.stack().reset_index()
    test_df.insert(loc=0, column='ID', value=test_df['Image'].astype(str) + "_" + test_df['Diagnosis'])
    test_df = test_df.drop(["Image", "Diagnosis"], axis=1)
    print("Saving submissions to submission.csv")
    test_df.to_csv('submission.csv', index=False)

    return test_df

def main():
    test_df = read_testset(testset_filename)
    test_generator = DataGenerator(test_df.index, None, 1, img_shape, test_images_dir)
    best_model = keras.models.load_model(args.model_filename, compile=False)
    create_submission(best_model, test_generator, test_df)


if __name__ == "__main__":
    main()