#!/usr/bin/env python
# coding: utf-8

import os
import sys
import ast
import random
import numpy as np
from PIL import Image
from scipy import ndimage, misc
from tensorflow import keras as K
import matplotlib.pylab as plt
import pandas as pd
import cv2
import pydicom
import data_flow


DATA_DIRECTORY = data_flow.TRAIN_DATA_PATH


class DataGenerator(K.utils.Sequence):
    """
    Generates data for Keras, including reading DICOM images and preprocessing/windowing images
    To resize images, set resize=True and pass new dims, e.g. dims=(256,256)
    To use the data loader for prediction/inference pass prediction=True, this will pass along
    a np.empty object for y, which is eventually discarded.

    Acceptable channel_types include: 'hu_norm','brain','subdural','soft_tissue',...
    Acceptable subtypes include: 'any','intraparenchymal','intraventricular','subarachoid','subdural','epidural'
    """
    def __init__(self,
                 csv_filename,
                 data_path,
                 batch_size=32,
                 channels = 3,
                 dims = (512,512),
                 num_classes = 2,
                 shuffle=True,
                 prediction=False,
                 augment=False,
                 subtype = "any",
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
        self.channel_types = channel_types
        self.subtype = subtype

        if self.subtype == "any":
            df_csv = pd.read_csv(csv_filename)
            df_subtype = df_csv[['id',self.subtype]]
            self.df = df_subtype.reset_index(drop=True)
        else:
            df_csv = pd.read_csv(csv_filename)
            df_subtype = df_csv[['id', self.subtype, 'any']]
            mask_df = df_subtype.loc[df_subtype['any'] == 1]
            mask_df.drop('column_name', axis=1, inplace=True)
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


    def rotate_img(self, img):
        degree = random.randrange(-10, 10)
        print("degree:",degree)
        return ndimage.rotate(img, degree, reshape=False)

    def flip_img(self, img):
        axis = random.choice([0, 1])
        return np.flip(img, axis=axis)
    
    def sp_noise(self, image, prob=0.025):
        '''
        Add salt and pepper noise to image
        prob: Probability of the noise
        '''
        output = np.zeros(image.shape,np.uint8)
        thres = 1 - prob 
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                rdn = random.random()
                if rdn < prob:
                    output[i][j] = 0
                elif rdn > thres:
                    output[i][j] = 255
                else:
                    output[i][j] = image[i][j]
        return output

    def augment_img(self, img):
        """
        Given a normalized and windowed 3 channel image, apply random augmentation
        """
        if random.choice([0, 1]) == 1:
            img = self.flip_img(img)
        if random.choice([0, 1]) == 1:
            img = self.rotate_img(img)
        if random.choice([0, 1]) == 1:
            img = self.sp_noise(img)
        return img


    def __data_generation(self, indexes):
        """
        Generates data containing batch_size samples
        """
        np.random.shuffle(self.indexes) #TODO DEBUG TAKE THIS OUT>>>>>>!!!!!!!!!!! debug
        batch_data = self.df.loc[indexes].values

        X = np.empty((self.batch_size, *self.dims, self.channels))
        y = np.empty((self.batch_size, self.num_classes))

        for idx in range(self.batch_size):
            filename = os.path.join(self.data_path, batch_data[idx][0])
            with pydicom.dcmread(filename) as ds:

                intercept, slope = self.hounsfield_translation(ds)
                img = ds.pixel_array#.astype(np.float)
                # img = np.array(img, dtype='uint8')

                channel_stack = []
                for channel_type in self.channel_types:
                    windowed_channel = self.window_image(img, intercept, slope, window_type=channel_type)

                    px = (img * slope + intercept).flatten()
                    plt.hist(px, bins=40);
                    plt.show()
                    # imgplot = plt.imshow(img, cmap=plt.cm.bone)
                    # plt.show()
                    # print(intercept,slope)
                    sys.exit()

                    
                    if self.dims != img.shape:
                        windowed_channel = cv2.resize(windowed_channel, self.dims, interpolation=cv2.INTER_LINEAR) #INTER_AREA
                    norm_channel = self.normalize_img(np.array(windowed_channel, dtype=float))

                    # imgplot = plt.imshow(norm_channel)
                    # plt.show()


                    channel_stack.append(norm_channel)
                
                rgb = np.dstack(channel_stack)
                if self.augment:
                    rgb = self.augment_img(rgb)
                
                # print(rgb[:,:,0].shape)
                # imgplot = plt.imshow(rgb[:,:,2])
                # plt.show()

                # add three channel image to batch index
                X[idx,] = rgb


            # If doing inference/prediction do not attempt to pass y value, leave as empty
            if not self.prediction:
                out = np.array([float(x) for x in batch_data[idx][1]], dtype='float')
                y[idx,] = out

        return X, y


if __name__ == "__main__":

    training_data = DataGenerator(csv_filename="./training.csv",
                                    data_path=DATA_DIRECTORY,
                                    num_classes=5,
                                    batch_size=1,
                                    augment=True,
                                    subtype = "any",
                                    channel_types = ['hu_norm','subdural','brain'])
    images, masks = training_data.__getitem__(1)

    # print(masks)
    # print(images)
