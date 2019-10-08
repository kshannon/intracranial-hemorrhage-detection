import sys
import os
import numpy as np
import datetime
import tensorflow as tf
from tensorflow import keras as K
# from keras_applications.resnet import ResNet50
from tensorflow import ConfigProto
from tensorflow import InteractiveSession
from data_loader import DataGenerator
import data_flow
import parse_config
from model_defs import *


if parse_config.USING_RTX_20XX:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))


MODEL_NAME = sys.argv[1]
DATA_DIRECTORY = data_flow.TRAIN_DATA_PATH
TRAIN_CSV = parse_config.VALIDATE_CSV
# TRAIN_CSV = "./class_one_training.csv"
VALIDATE_CSV = parse_config.VALIDATE_CSV
BATCH_SIZE = 1
EPOCHS = 5
RESIZE = (224,224) #comment out if not needed and erase param below
DIMS = (224,224)
TRAINING_DATA = DataGenerator(csv_filename=TRAIN_CSV, data_path=DATA_DIRECTORY, batch_size=BATCH_SIZE, resize=RESIZE, dims=DIMS)
VALIDATION_DATA = DataGenerator(csv_filename=VALIDATE_CSV, data_path=DATA_DIRECTORY, batch_size=BATCH_SIZE, resize=RESIZE, dims=DIMS)


# Saved models
checkpoint = K.callbacks.ModelCheckpoint(os.path.join('../models/', sys.argv[1] + '.pb'), verbose=1, save_best_only=True)
                                                       
# TensorBoard    
tb_logs = K.callbacks.TensorBoard(log_dir = os.path.join('tensorboards/', sys.argv[1]))
# log_dir="logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")


############## Specific Model Definition STARTS here ##############
num_chan_in = 2
height = 224
width = 224
num_classes = 6
bn_momentum = 0.99
BATCH_SIZE = 64
EPOCHS = 15 


inputs = K.layers.Input([height, width, num_chan_in], name="DICOM")

params = dict(kernel_size=(3, 3), activation="relu",
                      padding="same",
                      kernel_initializer="he_uniform")

img_1 = K.layers.BatchNormalization(momentum=bn_momentum)(inputs)
img_1 = K.layers.Conv2D(32, **params)(img_1)
img_1 = K.layers.MaxPooling2D(pool_size=(2,2))(img_1)

img_1 = K.layers.Conv2D(64, **params)((K.layers.BatchNormalization(momentum=bn_momentum))(img_1))
img_1 = K.layers.MaxPooling2D(name='skip1', pool_size=(2,2))(img_1)

# Residual block
img_2 = K.layers.Conv2D(128, **params) ((K.layers.BatchNormalization(momentum=bn_momentum))(img_1))
img_2 = K.layers.Conv2D(64, name='img2', **params) ((K.layers.BatchNormalization(momentum=bn_momentum))(img_2))
img_2 = K.layers.add( [img_1, img_2] )
img_2 = K.layers.MaxPooling2D(name='skip2', pool_size=(2,2))(img_2)

# Residual block
img_3 = K.layers.Conv2D(128, **params)((K.layers.BatchNormalization(momentum=bn_momentum))(img_2))
img_3 = K.layers.Conv2D(64, name='img3', **params)((K.layers.BatchNormalization(momentum=bn_momentum))(img_3))
img_res = K.layers.add( [img_2, img_3] )

# Filter residual output
img_res = K.layers.Conv2D(128, **params)((K.layers.BatchNormalization(momentum=bn_momentum))(img_res))

# Can you guess why we do this? Hint: Where did Flatten go??
img_res = K.layers.GlobalMaxPooling2D(name='global_pooling') ( img_res )

dense1 = K.layers.Dropout(0.5)(K.layers.Dense(256, activation = "relu")(img_res)) 
dense2 = K.layers.Dropout(0.5)(K.layers.Dense(64, activation = "relu")(dense1)) 
prediction = K.layers.Dense(num_classes, activation = 'sigmoid')(dense2)

model = K.models.Model(inputs=[inputs], outputs=[prediction])

opt = K.optimizers.Adam( lr = 1e-3, beta_1 = .9, beta_2 = .999, decay = 1e-3 )

model.compile(loss = K.losses.categorical_crossentropy, 
                optimizer = opt, 
                metrics = ['accuracy'])


############## Specific Model Definition ENDS here ##############
                                                                 
model.fit_generator(TRAINING_DATA, 
                    validation_data=VALIDATION_DATA, 
                    callbacks=[checkpoint, tb_logs],
                    epochs=EPOCHS)
