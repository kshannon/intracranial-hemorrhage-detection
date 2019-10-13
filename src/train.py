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
BATCH_SIZE = 32
EPOCHS = 15 
DIMS = (224,224) #512,512 default
RESIZE = True


training_data_gen = DataGenerator(csv_filename=TRAIN_CSV,
                                    data_path=DATA_DIRECTORY,
                                    batch_size=BATCH_SIZE,
                                    resize=RESIZE,
                                    dims=DIMS,
                                    augment=True)
validation_data_gen = DataGenerator(csv_filename=VALIDATE_CSV,
                                    data_path=DATA_DIRECTORY,
                                    batch_size=BATCH_SIZE,
                                    resize=RESIZE,
                                    dims=DIMS,
                                    augment=False)


#################################  CALLBACKS  ################################

# Saved models
checkpoint = K.callbacks.ModelCheckpoint(os.path.join('../models/', sys.argv[1] + '.pb'), verbose=1, save_best_only=True)
                                                       
# TensorBoard    
tb_logs = K.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR,
                                    update_freq='batch',
                                    profile_batch=0)

# Interrupt training if `val_loss` stops improving for over 2 epochs                                    
early_stop = K.callbacks.EarlyStopping(patience=2, monitor='val_loss')

# Stepped learning rate decay
lr_sched = callbks.step_decay_schedule(initial_lr=1e-4, decay_factor=0.75, step_size=2)


################################################################################# 
######################  YOUR MODEL DEFINITION GOES IN HERE  #####################
#################################################################################


num_chan_in = 3
height = 224
width = 224
num_classes = 6
bn_momentum = 0.99

# inputs = K.layers.Input([height, width, num_chan_in], name="DICOM")

resnet_model = ResNet50(input_shape=[height, width, num_chan_in],
                        weights='imagenet',
                        include_top=False,
                        pooling='avg',
                        utils = K.utils,
                        models = K.models,
                        layers = K.layers,
                        backend = K.backend)

# global_avg_pool = K.layers.GlobalAveragePooling2D(name='avg_pool')(resnet_model.output)
hemorrhage_output = K.layers.Dense(num_classes, activation="sigmoid", name='dense_output')(resnet_model.output)

model = K.models.Model(inputs=resnet_model.input, outputs=hemorrhage_output)

model.compile(loss=loss.weighted_log_loss,
                optimizer=K.optimizers.Adam(lr = 1e-3, beta_1 = .9, beta_2 = .999, decay = 1e-3),
                metrics=[loss.weighted_loss])

################################################################################# 
#######################  YOUR MODEL DEFINITION ENDs HERE  #######################
#################################################################################
#K.losses.categorical_crossentropy, 

# Here we go...
model.fit_generator(training_data_gen, 
                    validation_data=validation_data_gen, 
                    callbacks=[lr_sched, checkpoint, tb_logs, early_stop],
                    epochs=EPOCHS)

