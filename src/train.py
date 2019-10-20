#################    --INSTRUCTIONS TO RUN--   #################

# start training via:
# python train.py {model-name}-{dims}-{loss}-{subtype}-{monthDay-version}
# e.g.
# python train.py resnet50-dim224x224-bce-intraparenchymal-oct18v1

#################    --INSTRUCTIONS TO RUN--   #################



import sys
import os
import numpy as np
import datetime
import tensorflow as tf
from tensorflow import keras as K
from tensorflow import ConfigProto
from tensorflow import InteractiveSession
# from tensorflow.keras import metrics.CategoricalCrossentropy
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.applications import InceptionResNetV2

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
BATCH_SIZE = 32
EPOCHS = 20 
DIMS = (512,512)

params = dict(dims=DIMS,
          subtype="any",
          channel_types=['hu_norm','brain','soft_tissue'])

training_data_gen = DataGenerator(csv_filename=TRAIN_CSV,
                                    data_path=DATA_DIRECTORY,
                                    batch_size=BATCH_SIZE,
                                    augment=True,
                                    balance_data = True,
                                    **params)

validation_data_gen = DataGenerator(csv_filename=VALIDATE_CSV,
                                    data_path=DATA_DIRECTORY,
                                    batch_size=BATCH_SIZE,
                                    augment=False,
                                    balance_data = True,
                                    **params)


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

#CSV logging
csv_logger = K.callbacks.CSVLogger('../logs/' + sys.argv[1] + '.csv')



################################################################################# 
######################  YOUR MODEL DEFINITION GOES IN HERE  #####################
#################################################################################


num_chan_in = 3
height = DIMS[0]
width = DIMS[1]
num_classes = 2
bn_momentum = 0.99
kernel_initializer="he_uniform" #TODO: can we use this as a passed param to predefinned Keras models?

# inputs = K.layers.Input([height, width, num_chan_in], name="DICOM")
# image_input = Input(shape=(224, 224, 3))
# model = ResNet50(input_tensor=image_input, include_top=True,weights='imagenet')
# kernel_initializer="he_uniform" can we add this to pretrained mod

inceptionResnetV2_model = InceptionResNetV2(input_shape=[height, width, num_chan_in], 
                        include_top=False,
                        utils = K.utils,
                        models = K.models,
                        layers = K.layers,
                        backend = K.backend)
                        # weights='imagenet',
                        # pooling='avg' same thing as the layer below...

global_avg_pool = K.layers.GlobalAveragePooling2D(name='avg_pool')(inceptionResnetV2_model.output)
hemorrhage_output = K.layers.Dense(num_classes, activation="softmax", name='dense_output')(global_avg_pool)

model = K.models.Model(inputs=inceptionResnetV2_model.input, outputs=hemorrhage_output)

model.compile(loss=BinaryCrossentropy(),
                optimizer=K.optimizers.Adam(lr = 5e-4, beta_1 = .9, beta_2 = .999, decay = 0.8),
                metrics=[K.metrics.BinaryCrossentropy(), "accuracy"])#loss.weighted_loss

################################################################################# 
#######################  YOUR MODEL DEFINITION ENDs HERE  #######################
#################################################################################

# Here we go...
model.fit_generator(training_data_gen, 
                    validation_data=validation_data_gen, 
                    callbacks=[lr_sched, checkpoint, tb_logs, early_stop, csv_logger],
                    epochs=EPOCHS)