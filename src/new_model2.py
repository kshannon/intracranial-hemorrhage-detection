
test_images_dir = '../../data/stage_1_test_images/'
train_images_dir = '../../data/stage_1_train_images/'
trainset_filename = "../../data/stage_1_train.csv"
testset_filename = "../../stage_1_sample_submission.csv"
num_epochs = 10
image_shape = (512,512,3)
TRAINING = True # If False, then just load model and predict


import numpy as np
import pandas as pd
import pydicom
import os

from datetime import datetime

from math import ceil, floor, log
import cv2

import tensorflow as tf
from tensorflow import K as K


# from K_applications.resnet import ResNet50
from keras_applications.inception_v3 import InceptionV3
from keras_applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input

from sklearn.model_selection import ShuffleSplit


def correct_dcm(dcm):
    x = dcm.pixel_array + 1000
    px_mode = 4096
    x[x>=px_mode] = x[x>=px_mode] - px_mode
    dcm.PixelData = x.tobytes()
    dcm.RescaleIntercept = -1000

def window_image(dcm, window_center, window_width):

    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        correct_dcm(dcm)

    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)

    return img

def bsb_window(dcm):
    brain_img = window_image(dcm, 40, 80)
    subdural_img = window_image(dcm, 80, 200)
    soft_img = window_image(dcm, 40, 380)

    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380
    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1,2,0)

    return bsb_img

# %% [code]
def window_with_correction(dcm, window_center, window_width):
    if (dcm.BitsStored == 12) and (dcm.PixelRepresentation == 0) and (int(dcm.RescaleIntercept) > -100):
        correct_dcm(dcm)
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    return img

def window_without_correction(dcm, window_center, window_width):
    img = dcm.pixel_array * dcm.RescaleSlope + dcm.RescaleIntercept
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img = np.clip(img, img_min, img_max)
    return img

def window_testing(img, window):
    brain_img = window(img, 40, 80)
    subdural_img = window(img, 80, 200)
    soft_img = window(img, 40, 380)

    brain_img = (brain_img - 0) / 80
    subdural_img = (subdural_img - (-20)) / 200
    soft_img = (soft_img - (-150)) / 380
    bsb_img = np.array([brain_img, subdural_img, soft_img]).transpose(1,2,0)

    return bsb_img


# %% [code]
def _read(path, desired_size):
    """Will be used in DataGenerator"""

    dcm = pydicom.dcmread(path)

    try:
        img = bsb_window(dcm)
    except:
        img = np.zeros(desired_size)


    img = cv2.resize(img, desired_size[:2], interpolation=cv2.INTER_LINEAR)

    return img


# %% [code]
class DataGenerator(K.utils.Sequence):

    def __init__(self, list_IDs, labels=None, batch_size=1, img_size=(512, 512, 1),
                 img_dir=train_images_dir, *args, **kwargs):

        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.on_epoch_end()

    def __len__(self):
        return int(ceil(len(self.indices) / self.batch_size))

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]

        if self.labels is not None:
            X, Y = self.__data_generation(list_IDs_temp)
            return X, Y
        else:
            X = self.__data_generation(list_IDs_temp)
            return X

    def on_epoch_end(self):

        if self.labels is not None: # for training phase we undersample and shuffle
            # keep probability of any=0 and any=1
            keep_prob = self.labels.iloc[:, 0].map({0: 0.35, 1: 0.5})
            keep = (keep_prob > np.random.rand(len(keep_prob)))
            self.indices = np.arange(len(self.list_IDs))[keep]
            np.random.shuffle(self.indices)
        else:
            self.indices = np.arange(len(self.list_IDs))

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.img_size))

        if self.labels is not None: # training phase
            Y = np.empty((self.batch_size, 6), dtype=np.float32)

            for i, ID in enumerate(list_IDs_temp):
                X[i,] = _read(self.img_dir+ID+".dcm", self.img_size)
                Y[i,] = self.labels.loc[ID].values

            return X, Y

        else: # test phase
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = _read(self.img_dir+ID+".dcm", self.img_size)

            return X

# %% [markdown]
# ### 3a. loss function and metric


def weighted_log_loss(y_true, y_pred):
    """
    Can be used as the loss function in model.compile()
    ---------------------------------------------------
    """

    class_weights = np.array([2., 1., 1., 1., 1., 1.])

    eps = K.backend.epsilon()

    y_pred = K.backend.clip(y_pred, eps, 1.0-eps)

    out = -(         y_true  * K.backend.log(      y_pred) * class_weights
            + (1.0 - y_true) * K.backend.log(1.0 - y_pred) * class_weights)

    return K.backend.mean(out, axis=-1)


def _normalized_weighted_average(arr, weights=None):
    """
    A simple K implementation that mimics that of
    numpy.average(), specifically for this competition
    """

    if weights is not None:
        scl = K.backend.sum(weights)
        weights = K.backend.expand_dims(weights, axis=1)
        return K.backend.sum(K.backend.dot(arr, weights), axis=1) / scl
    return K.backend.mean(arr, axis=1)


def weighted_loss(y_true, y_pred):
    """
    Will be used as the metric in model.compile()
    ---------------------------------------------

    Similar to the custom loss function 'weighted_log_loss()' above
    but with normalized weights, which should be very similar
    to the official competition metric:
        https://www.kaggle.com/kambarakun/lb-probe-weights-n-of-positives-scoring
    and hence:
        sklearn.metrics.log_loss with sample weights
    """

    class_weights = K.backend.variable([2., 1., 1., 1., 1., 1.])

    eps = K.backend.epsilon()

    y_pred = K.backend.clip(y_pred, eps, 1.0-eps)

    loss = -(        y_true  * K.backend.log(      y_pred)
            + (1.0 - y_true) * K.backend/log(1.0 - y_pred))

    loss_samples = _normalized_weighted_average(loss, class_weights)

    return K.backend.mean(loss_samples)


def weighted_log_loss_metric(trues, preds):
    """
    Will be used to calculate the log loss
    of the validation set in PredictionCheckpoint()
    ------------------------------------------
    """
    class_weights = [2., 1., 1., 1., 1., 1.]

    epsilon = 1e-7

    preds = np.clip(preds, epsilon, 1-epsilon)
    loss = trues * np.log(preds) + (1 - trues) * np.log(1 - preds)
    loss_samples = np.average(loss, axis=1, weights=class_weights)

    return - loss_samples.mean()


class MyDeepModel:

    def __init__(self, engine, input_dims, batch_size=5, num_epochs=4, learning_rate=1e-3,
                 decay_rate=1.0, decay_steps=1, weights="imagenet", verbose=1):

        self.engine = engine
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.weights = weights
        self.verbose = verbose
        self.model_filename = '%s.hdf5' % engine.__name__
        self._build()

    def _build(self):


        engine = self.engine(include_top=False, weights=self.weights, input_shape=self.input_dims,
                             backend = K.backend, layers = K.layers,
                             models = K.models, utils = K.utils)

        x = K.layers.GlobalAveragePooling2D(name='avg_pool')(engine.output)
        x = K.layers.Dropout(0.5)(x)
        x = K.layers.Dense(K.backend.int_shape(x)[1], activation="relu", name="dense_hidden_1")(x)
        x = K.layers.Dropout(0.5)(x)
        x = K.layers.Dense(128, activation="relu", name="dense_hidden_2")(x)
        x = K.layers.Dropout(0.5)(x)
        out = K.layers.Dense(6, activation="sigmoid", name='dense_output')(x)

        self.model = K.models.Model(inputs=engine.input, outputs=out)

        self.model.compile(loss="binary_crossentropy", optimizer=K.optimizers.Adam(), metrics=["categorical_accuracy", "accuracy", weighted_loss])


    def fit_model(self, train_df, valid_df, test_df):

        # callbacks
        checkpointer = K.callbacks.ModelCheckpoint(filepath=self.model_filename, verbose=1, save_best_only=True)
        scheduler = K.callbacks.LearningRateScheduler(lambda epoch: self.learning_rate * pow(self.decay_rate, floor(epoch / self.decay_steps)))

        self.model.fit_generator(
            DataGenerator(
                train_df.index,
                train_df,
                self.batch_size,
                self.input_dims,
                train_images_dir
            ),
            epochs=self.num_epochs,
            verbose=self.verbose,
            validation_data=DataGenerator(
                valid_df.index,
                valid_df,
                self.batch_size,
                self.input_dims,
                train_images_dir
            ),
            validation_steps=1024,
            use_multiprocessing=True,
            workers=4,
            callbacks=[scheduler, checkpointer]
        )

    def save(self, path):
        self.model.save_weights(path)

    def load(self, path):
        self.model.load_weights(path)

# %% [markdown]
# ### 4. Read csv files
#

# %% [code]
def read_testset(filename="../../stage_1_sample_submission.csv"):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)

    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)

    return df

def read_trainset(filename="../../data/stage_1_train.csv"):
    df = pd.read_csv(filename)
    df["Image"] = df["ID"].str.slice(stop=12)
    df["Diagnosis"] = df["ID"].str.slice(start=13)

    duplicates_to_remove = [
        1598538, 1598539, 1598540, 1598541, 1598542, 1598543,
        312468,  312469,  312470,  312471,  312472,  312473,
        2708700, 2708701, 2708702, 2708703, 2708704, 2708705,
        3032994, 3032995, 3032996, 3032997, 3032998, 3032999
    ]

    df = df.drop(index=duplicates_to_remove)
    df = df.reset_index(drop=True)

    df = df.loc[:, ["Label", "Diagnosis", "Image"]]
    df = df.set_index(['Image', 'Diagnosis']).unstack(level=-1)

    return df


# %% [markdown]
# ### 6. Submit test predictions

def create_submission(model, data, test_df):

    pred = model.predict_generator(data, verbose=1)

    out_df = pd.DataFrame(pred, index=test_df.index, columns=test_df.columns)

    test_df = out_df.stack().reset_index()

    test_df.insert(loc=0, column='ID', value=test_df['Image'].astype(str) + "_" + test_df['Diagnosis'])

    test_df = test_df.drop(["Image", "Diagnosis"], axis=1)

    test_df.to_csv('submission.csv', index=False)

    return test_df


# obtain model
# model = MyDeepModel(engine=InceptionV3, input_dims=img_shape, batch_size=batch_size, learning_rate=1e-3,
#                     num_epochs=10, decay_rate=0.8, decay_steps=1, weights="imagenet", verbose=1)


model = MyDeepModel(engine=InceptionResNetV2, input_dims=img_shape, batch_size=batch_size, learning_rate=1e-3,
                    num_epochs=num_epochs, decay_rate=0.8, decay_steps=1, weights="imagenet", verbose=1)


if (TRAINING == True):

    df = read_trainset(trainset_filename)
    ss = ShuffleSplit(n_splits=10, test_size=0.1, random_state=42).split(df.index)
    # lets go for the first fold only
    train_idx, valid_idx = next(ss)

    # Train the model
    model.fit_model(df.iloc[train_idx], df.iloc[valid_idx])

    test_df = read_testset(testset_filename)
    test_generator = DataGenerator(test_df.index, None, 1, img_shape, test_images_dir)
    best_model = K.models.load_model(model.model_filename)
    prediction_df = create_submission(best_model, test_generator, test_df)

else:  # Prediction only

    test_df = read_testset(testset_filename)
    test_generator = DataGenerator(test_df.index, None, 1, img_shape, test_images_dir)
    best_model = K.models.load_model(model.model_filename)
    prediction_df = create_submission(best_model, test_generator, test_df)
