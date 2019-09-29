#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
from tensorflow import keras as K
import pandas as pd
import pydicom
import data_flow


DATA_DIRECTORY = data_flow.TRAIN_DATA_PATH #"../../stage_1_train_images/"


class DataGenerator(K.utils.Sequence):
    """
    Generates data for Keras
    """
    def __init__(self,
                 csv_filename,
                 data_path,
                 batch_size=32,
                 dims=(512,512),
                 channels=1,
                 num_classes=6,
                 shuffle=True):
        """
        Initialization
        """

        self.batch_size = batch_size
        self.dims = dims
        self.channels = channels
        self.num_classes = num_classes

        self.data_path = data_path

        self.df = pd.read_csv(csv_filename)
        self.indexes = np.arange(len(self.df))

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

    def __enter__(self):
        return self

    def __exit__(self, type, value, traceback):
        return isinstance(value, TypeError)

    def on_epoch_end(self):
        """
        Updates indexes after each epoch
        """
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def normalize_img(self, img):

        img = (img - np.mean(img)) / np.std(img)

        return img
        
        
    def window_img(self, img, min=-50, max=100):
       
        return self.normalize_img(np.clip(img, min, max))

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
              
                img = ds.pixel_array.astype(np.float)
                
                # If img not expected shape, then replace it with another image from dataset
                if (np.std(img) == 0) or (img.shape[0] != self.dims[0]) or (img.shape[1] != self.dims[1]):
                   print("Filename {} bad.".format(filename))
                   filename = os.path.join(self.data_path, batch_data[0][0])
                   # Create a new ds and img object
                   ds = pydicom.dcmread(filename)
                   img = ds.pixel_array.astype(np.float)
                
                # with a healthy img & ds we can get the windowing data
                window_center, window_width, intercept, slope = data_flow.get_windowing(ds)
                img = data_flow.window_image(ds.pixel_array, window_center, window_width, intercept, slope)
                X[idx,:,:,0] = self.normalize_img(np.array(img, dtype=float))
                X[idx,:,:,1] = self.window_img(img, -100, 100)
                
            y[idx,] = [float(x) for x in batch_data[idx][1][1:-1].split(" ")]

        return X, y


if __name__ == "__main__":

    training_data = DataGenerator(csv_filename="./training.csv", data_path=DATA_DIRECTORY)
    images, masks = training_data.__getitem__(1)

    print(masks)
