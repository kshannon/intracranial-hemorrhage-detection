import numpy as np
import tensorflow as tf
from tensorflow import keras as K
from data_loader import DataGenerator
import data_flow

from tensorflow import ConfigProto
from tensorflow import InteractiveSession
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
tf.keras.backend.set_session(tf.Session(config=config))


num_chan_in = 1
height = 512
width = 512
num_classes = 6

data_directory = data_flow.TRAIN_DATA_PATH #"../../stage_1_train_images/"

training_data = DataGenerator(csv_filename="./training.csv", data_path=data_directory)
validation_data = DataGenerator(csv_filename="./validation.csv", data_path=data_directory)

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
checkpoint = K.callbacks.ModelCheckpoint("saved_model.pb", verbose=1, save_best_only=True)
                                                       
# TensorBoard
tb_logs = K.callbacks.TensorBoard(log_dir="tensorboards")
                                                                                      
model.fit_generator(training_data, 
                    validation_data=validation_data, 
                    callbacks=[checkpoint, tb_logs],
                    epochs=10)
