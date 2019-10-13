import sys
import os
import numpy as np
import datetime
import tensorflow as tf
from tensorflow import keras as K
from tensorflow import ConfigProto
from tensorflow import InteractiveSession
# from tensorflow.keras import metrics.CategoricalCrossentropy
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications import ResNet50

import data_flow
import parse_config
from data_loader import DataGenerator
import custom_loss as loss
import custom_callbacks as callbks


if parse_config.USING_RTX_20XX:
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    tf.keras.backend.set_session(tf.Session(config=config))


MODEL_NAME = sys.argv[1]
DATA_DIRECTORY = data_flow.TRAIN_DATA_PATH
TRAIN_CSV = parse_config.TRAIN_CSV
VALIDATE_CSV = parse_config.VALIDATE_CSV
TENSORBOARD_DIR = os.path.join('tensorboards/', sys.argv[1])
CLASS_WEIGHTS = [1.0, 1.0, 1.0, 1.0, 1.0, 2.0]
BATCH_SIZE = 8
EPOCHS = 15

num_chan_in = 3
height = 512 #224
width = 512 #224

DIMS = (height,width) #512,512 default
RESIZE = True

num_classes = 1

training_data_gen = DataGenerator(csv_filename=TRAIN_CSV,
                                    data_path=DATA_DIRECTORY,
                                    batch_size=BATCH_SIZE,
                                    resize=RESIZE,
                                    dims=DIMS,
                                    num_classes=num_classes,
                                    augment=True,train=True,
                                    window=False)
validation_data_gen = DataGenerator(csv_filename=VALIDATE_CSV,
                                    data_path=DATA_DIRECTORY,
                                    batch_size=BATCH_SIZE,
                                    resize=RESIZE,
                                    num_classes=num_classes,
                                    dims=DIMS,
                                    augment=False,
                                    window=False)


#################################  CALLBACKS  ################################

# Saved models
checkpoint = K.callbacks.ModelCheckpoint(os.path.join('../models/', sys.argv[1] + '.pb'), verbose=1, save_best_only=True)

# TensorBoard
tb_logs = K.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR,
                                    update_freq='batch',
                                    profile_batch=0)

# Interrupt training if `val_loss` stops improving for over 2 epochs
early_stop = K.callbacks.EarlyStopping(patience=2, monitor='val_loss')

# Learning Rate
learning_rate = 5e-4
decay_rate = 0.8
decay_steps = 1
# lr_sched = callbks.step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, step_size=2)
lr_sched = K.callbacks.LearningRateScheduler(lambda epoch: learning_rate * pow(decay_rate, np.floor(epoch / decay_steps)))


#################################################################################
######################  YOUR MODEL DEFINITION GOES IN HERE  #####################
#################################################################################



bn_momentum = 0.99

# inputs = K.layers.Input([height, width, num_chan_in], name="DICOM")

resnet_model = ResNet50(input_shape=[height, width, num_chan_in],
                        weights='imagenet',
                        include_top=False,
                        utils = K.utils,
                        models = K.models,
                        layers = K.layers,
                        backend = K.backend)
                        # pooling='avg'

global_avg_pool = K.layers.GlobalAveragePooling2D(name='avg_pool')(resnet_model.output)
hemorrhage_output = K.layers.Dense(num_classes, activation="sigmoid", name='dense_output')(global_avg_pool)

model = K.models.Model(inputs=resnet_model.input, outputs=hemorrhage_output)

model.compile(loss="binary_crossentropy",
                optimizer=K.optimizers.Adam(lr = 5e-4, beta_1 = .9, beta_2 = .999, decay = 0.8),
                metrics=["accuracy"])

#################################################################################
#######################  YOUR MODEL DEFINITION ENDs HERE  #######################
#################################################################################
#K.losses.categorical_crossentropy,

# Here we go...
model.fit_generator(training_data_gen,
                    validation_data=validation_data_gen,
                    callbacks=[lr_sched, checkpoint, tb_logs, early_stop],
                    epochs=EPOCHS)
