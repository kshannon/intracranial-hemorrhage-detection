import numpy as np

import os

from datetime import datetime

from math import ceil, floor, log

import tensorflow as tf
import keras as K


from data_loader import DataGenerator

import pandas as pd


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
            + (1.0 - y_true) * K.backend.log(1.0 - y_pred))

    loss_samples = _normalized_weighted_average(loss, class_weights)

    return K.backend.mean(loss_samples)

class PredictionCheckpoint(K.callbacks.Callback):

    def on_epoch_end(self, epoch, logs={}):
        """
        Save each epoch file in case of crash
        """
        print("Saving checkpoint")
        self.model.save("epoch{}.hdf5".format(epoch))

class MyDeepModel:

    def __init__(self, engine, input_dims, batch_size=5, num_epochs=4, learning_rate=1e-3,
                 decay_rate=1.0, decay_steps=1, weights="imagenet", verbose=1, train_image_dir="", model_filename=""):

        self.engine = engine
        self.input_dims = input_dims
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.learning_rate = learning_rate
        self.decay_rate = decay_rate
        self.decay_steps = decay_steps
        self.weights = weights
        self.verbose = verbose
        self.model_filename = model_filename
        self.train_images_dir=train_image_dir
        self._build()

    def _build(self):


        engine = self.engine(include_top=False, weights=self.weights, input_shape=self.input_dims,
                             backend = K.backend, layers = K.layers,
                             models = K.models, utils = K.utils)

        x = K.layers.GlobalAveragePooling2D(name='avg_pool')(engine.output)
#         x = keras.layers.Dropout(0.2)(x)
#         x = keras.layers.Dense(keras.backend.int_shape(x)[1], activation="relu", name="dense_hidden_1")(x)
#         x = keras.layers.Dropout(0.1)(x)
        out = K.layers.Dense(6, activation="sigmoid", name='dense_output')(x)

        self.model = K.models.Model(inputs=engine.input, outputs=out)

        self.model.compile(loss="binary_crossentropy", optimizer=K.optimizers.Adam(), metrics=["categorical_accuracy", "accuracy", weighted_loss])

    def get_model_filename(self):

        return self.model_filename

    def fit_model(self, train_df, valid_df):

        # callbacks
        checkpointer = K.callbacks.ModelCheckpoint(filepath=self.model_filename, verbose=1, save_best_only=True)
        scheduler = K.callbacks.LearningRateScheduler(lambda epoch: self.learning_rate * pow(self.decay_rate, floor(epoch / self.decay_steps)))

        self.model.fit_generator(
            DataGenerator(
                train_df.index,
                train_df,
                self.batch_size,
                self.input_dims,
                self.train_images_dir
            ),
            epochs=self.num_epochs,
            verbose=self.verbose,
            validation_data=DataGenerator(
                valid_df.index,
                valid_df,
                self.batch_size,
                self.input_dims,
                self.train_images_dir
            ),
            use_multiprocessing=True,
            workers=4,
            callbacks=[scheduler, checkpointer, PredictionCheckpoint()]
        )

    def save(self, path):
        self.model.save(path)

    def load(self, path):
        self.model.load_weights(path)


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
