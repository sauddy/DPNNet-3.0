# import the necessary packages

import pandas as pd
import numpy as np
import glob
import cv2
import os
import matplotlib.pyplot as plt
# TRAIN_TEST_SPLIT = 0.7
print("We are importing the generator module")
# from data_aug import data_man as d
class Disk_planet_generator():

    '''
    ## This class will be used to train the CNN and the hybrid CNN_DPPNet 
       Input: A dataframe with all the data along with the image path
       Output : A generator to get images according to batch size for training

    '''

    def __init__(self, df):  # getting the comple dataset
        self.df = df

    def generate_split_indexes(self, test_size,Validation_split):
        '''
        Input: A complete csv with data and image path
        Output: index of train, validation and test after a random shuffle
                the divsion is made depending in the Train_test_split value

        '''
        print("[INFO] Splitting the data into train, val and test set...")
        p = np.random.permutation(len(self.df)) ## ramdomising the data
        # p = np.arange(len(self.df)) ## keeping the ordered data
        TRAIN_TEST_SPLIT = 1 - test_size
        train_up_to = int(len(self.df) * TRAIN_TEST_SPLIT)
        train_idx = p[:train_up_to]
        test_idx = p[train_up_to:]
        train_up_to = int(train_up_to * (1-Validation_split))
        train_idx, valid_idx = train_idx[:train_up_to], train_idx[train_up_to:]

        return train_idx, valid_idx, test_idx

    @staticmethod  # using the static method decorator (no use of self in this method)
    def process_disk_images(imagePath, X_res, Y_res):

        image = cv2.imread(imagePath)  # read the image corresponding to the path
        # cropping the image
        left = 44
        top = 44
        right = 556
        bottom = 556
        crop_image = image[left:right, top:bottom]
        crop_image = cv2.resize(crop_image, (X_res, Y_res))  # downsizing/resizing the image
        im = crop_image / 255.0

        return np.array(im)

    def generate_images(self, image_idx, is_training, batch_size, X_res, Y_res):

        # arrays to store our batched data
        # images, disk_params, appended_disk_params = [], [], []
        images, labels_array = [], []

        while True:
            for idx in image_idx:
                # readind a row from complete dataframe corresponding to a index
                data = self.df.iloc[idx].to_frame().transpose()

                # loading each image
                image_path = data.image_path.to_list()  # image path
                # print(image_path, idx)
                image = self.process_disk_images(image_path[0], X_res, Y_res)  # load image
                images.append(image)

                # target variable, i.e. planet mass for the traning purpose

                labels = data.pop("Planet_Mass1").to_numpy()
                labels_array.append(labels)
                # # corresposding disk properties minus the imgae path
                # disk_params = data.drop(columns=['file'])  # dropping the image path column
                # appended_disk_params.append(disk_params)

                # yielding condition
                if len(images) >= batch_size:  # astype(np.float32) was added to solve the error a NumPy array to a Tensor
                    yield (np.array(images).astype(np.float64), np.array(labels_array).astype(np.float64))
                    # yield np.array(images), np.array(labels_array)
                    # yield np.stack(images, axis=0), np.stack(labels_array, axis=0)
                    # , pd.concat(appended_disk_params, ignore_index=True, axis=0) np.stack(labels_array, axis=0)
                    images, labels_array = [], []
            if not is_training:
                break
        # appended_disk_params = pd.concat(appended_disk_params, ignore_index=True, axis=0)
        # return appended_disk_params, images
