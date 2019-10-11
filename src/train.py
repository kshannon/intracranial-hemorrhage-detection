import sys
import os
import numpy as np
import datetime
import tensorflow as tf
from tensorflow import keras as K
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
VALIDATE_CSV = parse_config.VALIDATE_CSV
TENSORBOARD_DIR = os.path.join('tensorboards/', sys.argv[1])
CLASS_WEIGHTS = [0.16, 0.16, 0.16, 0.16, 0.16, 0.2]
BATCH_SIZE = 16
EPOCHS = 15 
DIMS = (512,512)
# RESIZE = (224,224) #comment out if not needed and erase param below


training_data_gen = DataGenerator(csv_filename=TRAIN_CSV,
                                    data_path=DATA_DIRECTORY,
                                    batch_size=BATCH_SIZE,
                                    resize=None,
                                    dims=DIMS)
validation_data_gen = DataGenerator(csv_filename=VALIDATE_CSV,
                                    data_path=DATA_DIRECTORY,
                                    batch_size=BATCH_SIZE,
                                    resize=None,
                                    dims=DIMS)


######################  CALLBACKS  #####################

# Saved models
checkpoint = K.callbacks.ModelCheckpoint(os.path.join('../models/', sys.argv[1] + '.pb'), verbose=1, save_best_only=True)
                                                       
# TensorBoard    
tb_logs = K.callbacks.TensorBoard(log_dir=TENSORBOARD_DIR,
                                    update_freq='batch')

# Interrupt training if `val_loss` stops improving for over 2 epochs                                    
early_stop = K.callbacks.EarlyStopping(patience=2, monitor='val_loss')


######################  CUSTOM LOSS FUNCTIONS  ##################### 
# multilabel loss (optional weighted)
def multilabel_loss(class_weights=None):
    def multilabel_loss_inner(y_true, logits):
        logits = tf.cast(logits, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        
        # compute single class cross entropies:
        contributions = tf.maximum(logits, 0) - tf.multiply(logits, y_true) + tf.log(1.+tf.exp(-tf.abs(logits)))

        # contributions have shape (n_samples, n_classes), we need to reduce with mean over samples to obtain single class xentropies:
        single_class_cross_entropies = tf.reduce_mean(contributions, axis=0)

        # if None, weight equally:
        if class_weights is None:
            loss = tf.reduce_mean(single_class_cross_entropies)
        else:
            weights = tf.constant(class_weights, dtype=tf.float32)
            loss = tf.reduce_sum(tf.multiply(weights, single_class_cross_entropies))
        return loss
    return multilabel_loss_inner


################################################################################# 
######################  YOUR MODEL DEFINITION GOES IN HERE  #####################
#################################################################################

num_chan_in = 3
height = 512
width = 512
num_classes = 6
bn_momentum = 0.99

inputs = K.layers.Input([height, width, num_chan_in], name="DICOM")

params = dict(kernel_size=(3, 3),
                activation="relu",
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

# Tendancy to flatten
img_res = K.layers.GlobalMaxPooling2D(name='global_pooling') ( img_res )

dense1 = K.layers.Dropout(0.5)(K.layers.Dense(256, activation = "relu")(img_res)) 
dense2 = K.layers.Dropout(0.5)(K.layers.Dense(64, activation = "relu")(dense1)) 
prediction = K.layers.Dense(num_classes, activation = 'sigmoid')(dense2)

model = K.models.Model(inputs=[inputs], outputs=[prediction])

opt = K.optimizers.Adam(lr = 1e-3, beta_1 = .9, beta_2 = .999, decay = 1e-3)

model.compile(loss = K.losses.categorical_crossentropy, 
                optimizer = opt, 
                metrics = [multilabel_loss(class_weights=CLASS_WEIGHTS)])

################################################################################# 
#######################  YOUR MODEL DEFINITION ENDs HERE  #######################
#################################################################################


# Here we go...
model.fit_generator(training_data_gen, 
                    validation_data=validation_data_gen, 
                    callbacks=[checkpoint, tb_logs, early_stop],
                    epochs=EPOCHS)
