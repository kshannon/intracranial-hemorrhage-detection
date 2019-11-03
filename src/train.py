from datetime import datetime
from model import MyDeepModel, create_submission
from data_loader import read_testset, read_trainset, DataGenerator

import keras as K

from sklearn.model_selection import ShuffleSplit


# from K_applications.resnet import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from keras.applications.densenet import DenseNet121
from keras.applications.mobilenet_v2 import MobileNetV2

test_images_dir = '../../data/stage_1_test_images/'
train_images_dir = '../../data/stage_1_train_images/'
trainset_filename = "../../data/stage_1_train.csv"
testset_filename = "../../stage_1_sample_submission.csv"
num_epochs = 10
img_shape = (256,256,3)
batch_size=32
TRAINING = True # If False, then just load model and predict

engine=InceptionV3
model_filename="InceptionV3_{}.hdf5".format(datetime.now().strftime('%Y_%m_%d_%H_%M_%S'))
#model_filename="wrapper_2019_11_02_22_06_45.hdf5"

# obtain model
model = MyDeepModel(engine=engine, input_dims=img_shape, batch_size=batch_size,
                    learning_rate=5e-4,
                    num_epochs=num_epochs, decay_rate=0.8,
                    decay_steps=1,
                    weights="imagenet", verbose=1,
                    train_image_dir=train_images_dir,
                    model_filename=model_filename)


#model.load(model_filename)  # Use previous checkpoint


if (TRAINING == True):

    df = read_trainset(trainset_filename)
    ss = ShuffleSplit(n_splits=10, test_size=0.1, random_state=816).split(df.index)
    # lets go for the first fold only
    train_idx, valid_idx = next(ss)

    # Train the model
    model.fit_model(df.iloc[train_idx], df.iloc[valid_idx])


test_df = read_testset(testset_filename)
test_generator = DataGenerator(test_df.index, None, 1, img_shape, test_images_dir)
best_model = K.models.load_model(model.model_filename, compile=False)

prediction_df = create_submission(best_model, test_generator, test_df)
