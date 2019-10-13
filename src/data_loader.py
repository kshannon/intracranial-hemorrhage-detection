#!/usr/bin/env python
# coding: utf-8

import os
import random
import numpy as np
from tensorflow import keras as K
import pandas as pd
import cv2
import pydicom
import data_flow


DATA_DIRECTORY = data_flow.TRAIN_DATA_PATH


class DataGenerator(K.utils.Sequence):
    """
    Generates data for Keras, including reading DICOM images and preprocessing/windowing images
    To resize images, set resize=True and pass new dims, e.g. dims=(256,256)
    To use the data lloader for prediction/inference pass prediction=True, this will pass along 
    a np.empty object for y, which is eventually discarded.
    """
    def __init__(self,
                 csv_filename,
                 data_path,
                 batch_size=32,
                 dims=(512,512),
                 channels=3,
                 num_classes=6,
                 shuffle=True,
                 prediction=False,
                 resize=False,
                 window=False,
                 augment=False):
        """
        Class attribute initialization
        """
        self.batch_size = batch_size
        self.dims = dims
        self.channels = channels
        self.num_classes = num_classes
        self.data_path = data_path
        self.df = pd.read_csv(csv_filename, header=None)
        self.indexes = np.arange(len(self.df))
        self.window = window
        self.resize = resize
        self.prediction = prediction
        self.augment = augment
        self.shuffle = shuffle
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


    def window_image(self, img, window_center, window_width, intercept, slope):
        """
        Given a CT scan img apply a windowing to the image
        Arguments:
            img {np.array} -- array of a dicom img processed by pydicom.dcmread()
            window_center,window_width,intercept,slope {floats} -- values provided by dicom file metadata
        Attribution: This code comes from Richard McKinley's Kaggle kernel
        """
        img = (img * slope + intercept)
        img_min = window_center - window_width // 2
        img_max = window_center + window_width // 2
        img[img < img_min] = img_min
        img[img > img_max] = img_max
        return img


    def rotate_img(self, X):
        return X
    
    
    def flip_img(self, X):
        return X


    def augment_img(self, X):
        """
        Given a normalized and windowed 3 channel image, apply random augmentation
        """
        #TODO: add random scheme for augmentation selection
        X = self.flip_img(X)
        X = self.rotate_img(X)
        return X


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
              
                intercept, slope = self.hounsfield_translation(ds)
                img = ds.pixel_array.astype(np.float)
                img = np.array(img, dtype='uint8')

                if self.window:
                    tissue_window = self.window_image(img, 40, 40, intercept, slope)
                    brain_window = self.window_image(img, 50, 100, intercept, slope)
                    blood_window = self.window_image(img, 60, 40, intercept, slope)
                    if self.resize:
                        tissue_window = cv2.resize(tissue_window, self.dims, interpolation = cv2.INTER_AREA)
                        brain_window = cv2.resize(brain_window, self.dims, interpolation = cv2.INTER_AREA)
                        blood_window = cv2.resize(blood_window, self.dims, interpolation = cv2.INTER_AREA)

                    X[idx,:,:,0] = self.normalize_img(np.array(tissue_window, dtype=float))
                    X[idx,:,:,1] = self.normalize_img(np.array(brain_window, dtype=float))
                    X[idx,:,:,2] = self.normalize_img(np.array(blood_window, dtype=float))

                if not self.window:
                    img = (img * slope + intercept)
                    #TODO: just make this check if img size is == (512,512) get rid of resize=True attribute
                    if self.resize:
                        img = cv2.resize(img, self.dims, interpolation=cv2.INTER_LINEAR)
                    X[idx,] = np.stack((self.normalize_img(np.array(img, dtype=float)),)*3, axis=-1)

                # data augmentation gauntlet
                if self.augment:
                    X = self.augment_img(X)

            # If doing inference/prediction do not attempt to pass y value, leave as empty
            if not self.prediction:    
                y[idx,] = [float(x) for x in batch_data[idx][1][1:-1].split(" ")]
        
        
        return X, y


if __name__ == "__main__":

    training_data = DataGenerator(csv_filename="./training.csv", data_path=DATA_DIRECTORY)
    images, masks = training_data.__getitem__(1)

    print(masks)
