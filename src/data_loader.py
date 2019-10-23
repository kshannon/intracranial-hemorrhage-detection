#!/usr/bin/env python
# coding: utf-8

import os
import sys
import ast
import random
from PIL import Image
import numpy as np
from scipy import ndimage
from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter
import pandas as pd
import cv2
import pydicom
from tensorflow import keras as K
import data_flow

from tqdm import tqdm
from multiprocessing import  Pool


# debuggin
# import matplotlib.pylab as plt

np.seterr(invalid='raise', divide='raise')
DATA_DIRECTORY = data_flow.TRAIN_DATA_PATH


class DataGenerator(K.utils.Sequence):
    """
    Generates data for Keras, including reading DICOM images and preprocessing/windowing images
    To resize images, set dims to a new tuple of ints e.g. (244,244)
    To use the data loader for prediction/inference pass prediction=True, this will pass along
    a np.empty object for y, which is eventually discarded.
    Setting augment=True will randomly flip, rotate (+/-10), and add salt&pepper noise to the three channel imgs.
    Use the subtype argument to tell the network which subtype to train for. Each one trains binary cross entropy.
    'any' will train on all data, anyother subtype will train only on IH - positive data, where class 1 is the chosen
    subtype and all other subtypes are now labeled class 0. We can train this way because each real subtype has mutually
    exclusive probability.

    Acceptable strings for applying windowing are channel_types='' ['hu_norm',
                                                                    'brain',
                                                                    'subdural',
                                                                    'soft_tissue']

    Acceptable strings for subtype='' ['any',
                                        'intraparenchymal',
                                        'intraventricular',
                                        'subarachnoid',
                                        'subdural',
                                        'epidural'
    """
    def __init__(self,
                 csv_filename,
                 data_path,
                 batch_size = 32,
                 channels = 3,
                 dims = (512,512),
                 num_classes = 1,
                 shuffle = True,
                 prediction = False,
                 augment = False,
                 sigmoid = False,
                 subtype = "any",
                 balance_data = True,
                 channel_types = ['hu_norm','hu_norm','hu_norm']):
        """
        Class attribute initialization
        """
        self.data_path = data_path
        self.batch_size = batch_size
        self.channels = channels
        self.dims = dims
        self.num_classes = num_classes
        self.prediction = prediction
        self.augment = augment
        self.shuffle = shuffle
        self.sigmoid = sigmoid
        self.channel_types = channel_types
        self.subtype = subtype
        self.balance_data = balance_data

        if self.prediction:
            self.df = pd.read_csv(csv_filename)
        else:
            if self.subtype == "any":
                df_csv = pd.read_csv(csv_filename)
                df_subtype = df_csv[['filename',self.subtype]]
                df_subtype = df_subtype[df_subtype.filename != 'ID_470e639ae.dcm']
                self.df = df_subtype.reset_index(drop=True)
            else:
                if self.balance_data:
                    df_csv = pd.read_csv(csv_filename)
                    df_subtype = df_csv[['filename', self.subtype, 'any']]
                    mask_df = df_subtype.loc[df_subtype['any'] == 1]
                    mask_df.drop('any', axis=1, inplace=True)

                    #now we need to balance the data for class 1/0
                    class1_df = mask_df.loc[mask_df[self.subtype] == 1]
                    class0_df = mask_df.loc[mask_df[self.subtype] == 0]

                    # randomly subset class 0 to match class 1 50/50
                    class0_df = class0_df.sample(frac=1, random_state=13).reset_index(drop=True)
                    class0_df = class0_df.sample(n=class1_df.shape[0], random_state=13)

                    # Reconstitute balanced dataset, shuffle whole dataset
                    balanced_df = pd.concat([class1_df, class0_df], ignore_index=True)
                    balanced_df = balanced_df[balanced_df.filename != 'ID_470e639ae.dcm']
                    balanced_df = balanced_df.sample(frac=1, random_state=13).reset_index(drop=True)
                    self.df = balanced_df

                if not self.balance_data:
                    # here we choose not to balance the dataset between class 1/0
                    df_csv = pd.read_csv(csv_filename)
                    df_subtype = df_csv[['filename', self.subtype, 'any']]
                    mask_df = df_subtype.loc[df_subtype['any'] == 1]
                    mask_df.drop('any', axis=1, inplace=True)
                    mask_df = mask_df[mask_df.filename != 'ID_470e639ae.dcm']
                    self.df = mask_df.reset_index(drop=True)

        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()

    def __len__(self):
        """
        Denotes the number of batches per epoch
        """
        return len(self.df) // self.batch_size

    def __getitem__(self, index):
        """
        Generate one batch of data
        """
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]
        # Generate data
        X, y = self.__data_generation(indexes)
        return X, y

    def on_epoch_end(self):
        """
        Updates (shuffles) indexes after each epoch
        """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def normalize_img(self, img):
        """
        Return normalized numpy img array, plain & simple
        """
        # img = (img - np.mean(img)) / np.std(img)
        return 2 * (img - img.min())/(img.max() - img.min()) - 1

    def hounsfield_translation(self, data):
        """
        Retrieves windowing data from dicom metadata
        Arguments:
            data {pydicom metadata obj} -- object returned from pydicom dcmread()
        Attribution: This code inspired from Richard McKinley's Kaggle kernel
        """
        if type(data.RescaleIntercept) == pydicom.multival.MultiValue:
            intercept = int(data.RescaleIntercept[0])
        else:
            intercept = int(data.RescaleIntercept)

        if type(data.RescaleSlope) == pydicom.multival.MultiValue:
            slope = int(data.RescaleSlope[0])
        else:
            slope = int(data.RescaleSlope)

        return intercept, slope

    def window_image(self, img, intercept, slope, window_type):
        """
        Given a CT scan img apply types of windowing
        Attribution: Part of this code comes from Richard McKinley's Kaggle kernel
        """
        window_value = {'hu_norm':None, 'subdural':[50,130], 'brain':[50,100], 'soft_tissue':[0,350]} # [window_center, window_width]
        img = (img * slope + intercept)

        if window_type == 'hu_norm':
            return img

        if window_type == 'subdural':
            window_center = window_value['subdural'][0]
            window_width = window_value['subdural'][1]
        elif window_type == 'brain':
            window_center = window_value['brain'][0]
            window_width = window_value['brain'][1]
        elif window_type == 'soft_tissue':
            window_center = window_value['soft_tissue'][0]
            window_width = window_value['soft_tissue'][1]
        else:
            return img

        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img[img < img_min] = img_min
        img[img > img_max] = img_max
        return img

    def sigmoid_window(self, img, window_center, window_width, filename, U=1.0, eps=(1.0 / 255.0)):
        intercept, slope = self.hounsfield_translation(img)
        img = img.pixel_array * slope + intercept
        ue = np.log((U / eps) - 1.0)
        W = (2 / window_width) * ue
        b = ((-2 * window_center) / window_width) * ue
        z = W * img + b
        try:
            img = U / (1 + np.power(np.e, -1.0 * z))
            img = (img - np.min(img)) / (np.max(img) - np.min(img))
        except FloatingPointError as error:
            print('#'*50)
            print(error)
            print('filename: ' + filename)
            print('#'*50)
        return img

    def sigmoid_bsb_window(self, img, filename):
        '''
        Attribution: Sigmoid & BSB img code/idea comes from Ryan Epp's
        https://www.kaggle.com/reppic/gradient-sigmoid-windowing/notebook awesome kernel.
        '''
        brain_img = self.sigmoid_window(img, 40, 80, filename)
        subdural_img = self.sigmoid_window(img, 80, 200, filename)
        bone_img = self.sigmoid_window(img, 600, 2000, filename)

        if self.dims != brain_img.shape:
            brain_img = cv2.resize(brain_img, self.dims, interpolation=cv2.INTER_LINEAR)
        if self.dims != subdural_img.shape:
            subdural_img = cv2.resize(subdural_img, self.dims, interpolation=cv2.INTER_LINEAR)
        if self.dims != bone_img.shape:
            bone_img = cv2.resize(bone_img, self.dims, interpolation=cv2.INTER_LINEAR)

        bsb_img = np.zeros((brain_img.shape[0], brain_img.shape[1], 3))
        bsb_img[:, :, 0] = brain_img
        bsb_img[:, :, 1] = subdural_img
        bsb_img[:, :, 2] = bone_img
        return bsb_img

    def rotate_img(self, img):
        degree = random.randrange(-10, 10)
        return ndimage.rotate(img, degree, reshape=False)

    def flip_img(self, img):
        axis = random.choice([0, 1])
        return np.flip(img, axis=axis)

    def sp_noise(self, image, prob):
        '''
        Add salt and pepper noise to image
        prob: Probability of the noise
        '''
        thres = 1 - prob
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    image[i][j] = 0
                elif rdn > thres:
                    image[i][j] = 255
                else:
                    continue
        return image

    def elastic_transform(self, image, alpha, sigma, random_state=None):
        """Elastic deformation of images as described in [Simard2003]_.
        .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
        Convolutional Neural Networks applied to Visual Document Analysis", in
        Proc. of the International Conference on Document Analysis and
        Recognition, 2003.-https://gist.github.com/erniejunior/601cdf56d2b424757de5
        """
        if random_state is None:
            random_state = np.random.RandomState(None)

        shape = image.shape
        dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma, mode="constant", cval=0) * alpha
        dz = np.zeros_like(dx)

        x, y, z = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]), np.arange(shape[2]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1)), np.reshape(z, (-1, 1))

        distored_image = map_coordinates(image, indices, order=1, mode='reflect')
        return distored_image.reshape(image.shape)

    def crop_img(self, img):
        pass

    def augment_img(self, img):
        """
        Given a normalized and windowed 3 channel image, apply random augmentation
        """
        if random.choice([0, 1]) == 1:
            img = self.flip_img(img)
        if random.choice([0, 1]) == 1:
            img = self.rotate_img(img)
        if random.choice([0, 1]) == 1:
            img = self.sp_noise(img, prob=0.0075)
        if random.choice([0, 1]) == 1:
            img = self.elastic_transform(img,alpha=400,sigma=8)
        return img

    def __data_generation(self, indexes):
        """
        Generates data containing batch_size samples
        """
        batch_data = self.df.loc[indexes].values

        X = np.empty((self.batch_size, *self.dims, self.channels))
        y = np.empty((self.batch_size, self.num_classes))

        for idx in range(self.batch_size):
            filename = os.path.join(self.data_path, batch_data[idx][0])
            with pydicom.dcmread(filename) as ds:

                # Use a sigmoid windowing scheme with brain, subdural, bone windows
                if self.sigmoid:
                    rgb = self.sigmoid_bsb_window(ds, filename)
                    if self.augment:
                        rgb = self.augment_img(rgb)
                    X[idx,] = rgb

                # Use a traditional non sigmoid windowing scheme, windows passed as args to the class initializer
                if not self.sigmoid:
                    intercept, slope = self.hounsfield_translation(ds)
                    img = ds.pixel_array #.astype('float32') #astype(ds.pixel_array.dtype)
                    # img = np.array(img, dtype=np.float) # Real point of contention here....dtype='uint8' looked really weird

                    channel_stack = []
                    for channel_type in self.channel_types:
                        windowed_channel = self.window_image(img, intercept, slope, window_type=channel_type)

                        if self.dims != img.shape:
                            windowed_channel = cv2.resize(windowed_channel, self.dims, interpolation=cv2.INTER_LINEAR) #vs. INTER_LINEAR...?

                        norm_channel = self.normalize_img(np.array(windowed_channel, dtype=float))
                        channel_stack.append(norm_channel)

                    rgb = np.dstack(channel_stack)
                    if self.augment:
                        rgb = self.augment_img(rgb)

                    # add three channel "rgb" image to batch's index. rinse & repeat.
                    X[idx,] = rgb


            # If doing inference/prediction do not attempt to pass y value, leave as empty
            if not self.prediction:
                out = np.array(batch_data[idx][1], dtype='float')
                y[idx,] = out

        return X, y


if __name__ == "__main__":

    training_data = DataGenerator(csv_filename="./training.csv",
                                    data_path=DATA_DIRECTORY,
                                    batch_size=32,
                                    augment=True,
                                    dims=(224,224),
                                    sigmoid = True,
                                    subtype = "any",
                                    channel_types = ['subdural','soft_tissue','brain'])
    # images, masks = training_data.__getitem__(1)
    for i in tqdm(range(4368)):
        images, masks = training_data.__getitem__(i)
        # print('batch good')

    # print(masks)
    # print(images)



# ID_470e639ae.dcm



     # DEBUGGING - plot images after windowing/HU_norm
                    # print(X.shape)
                    # imgplot = plt.imshow(rgb)
                    # r = plt.imshow(rgb[:,:,0])
                    # g = plt.imshow(rgb[:,:,1])
                    # b = plt.imshow(rgb[:,:,2])
                    # plt.show()
                    # sys.exit()
                     # DEBUGGING - plot images after windowing/HU_norm
                    # imgplot = plt.imshow(rgb, cmap=plt.cm.bone)
                    # plt.show()
