import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from tensorflow import ConfigProto
from tensorflow import InteractiveSession
from data_loader import DataGenerator
import data_flow
import parse_config

if parse_config.USING_RTX_20XX:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))

DATA_DIRECTORY = data_flow.TRAIN_DATA_PATH
TRAIN_CSV = parse_config.VALIDATE_CSV
VALIDATE_CSV = parse_config.VALIDATE_CSV

num_chan_in = 2
height = 512
width = 512
num_classes = 6


training_data = DataGenerator(csv_filename=TRAIN_CSV, data_path=DATA_DIRECTORY)
validation_data = DataGenerator(csv_filename=VALIDATE_CSV, data_path=DATA_DIRECTORY)

inputs = K.layers.Input([height, width, num_chan_in], name="DICOM")

params = dict(kernel_size=(3, 3), activation="relu",
                      padding="same",
                      kernel_initializer="he_uniform")

convA = K.layers.Conv2D(name="convAa", filters=32, **params)(inputs)
convA = K.layers.Conv2D(name="convAb", filters=32, **params)(convA)
poolA = K.layers.MaxPooling2D(name="poolA", pool_size=(2, 2))(convA)

convB = K.layers.Conv2D(name="convBa", filters=64, **params)(poolA)
convB = K.layers.Conv2D(
    name="convBb", filters=64, **params)(convB)
poolB = K.layers.MaxPooling2D(name="poolB", pool_size=(2, 2))(convB)

convC = K.layers.Conv2D(name="convCa", filters=32, **params)(poolB)
convC = K.layers.Conv2D(
    name="convCb", filters=32, **params)(convC)
poolC = K.layers.MaxPooling2D(name="poolC", pool_size=(2, 2))(convC)

flat = K.layers.Flatten()(poolC)

drop = K.layers.Dropout(0.5)(flat)

dense1 = K.layers.Dense(128, activation="relu")(drop)

dense2 = K.layers.Dense(num_classes, activation="sigmoid")(dense1)

model = K.models.Model(inputs=[inputs], outputs=[dense2])

opt = K.optimizers.Adam()

model.compile(loss=K.losses.categorical_crossentropy,
              optimizer=opt,
              metrics=['accuracy'])


# Saved models
checkpoint = K.callbacks.ModelCheckpoint("../models/baseline-model.pb", verbose=1, save_best_only=True)
                                                       
# TensorBoard
tb_logs = K.callbacks.TensorBoard(log_dir="tensorboards")
                                                                                      
model.fit_generator(training_data, 
                    validation_data=validation_data, 
                    callbacks=[checkpoint, tb_logs],
                    epochs=1)
