import numpy as np
import keras
import os
from tifffile import imread

class DataGenerator(keras.utils.all_utils.Sequence):

    def __init__(self, partition, batch_size, img_size=(256,256,64,3), mask_size=(256,256,64,2), shuffle=True):
        
        self.partition = partition
        self.list_IDs = sorted(os.listdir('./Dataset/Patches_Pre_64/'+ partition +'/Images/'), key=self.order_dirs)

        self.dim = img_size
        self.mask_dim = mask_size
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.list_IDs))

    def __len__(self):
        'Denotes the number of batches per epoch'
        return len(self.list_IDs) // self.batch_size

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, mask = self.__get_data(list_IDs_temp)

        return X, mask

    def __get_data(self, list_IDs_temp):
        # Initialization
        X = np.empty((self.batch_size, *self.dim))
        mask = np.empty((self.batch_size, *self.mask_dim))

        # Generate data
        for i, ID_path in enumerate(list_IDs_temp):

            X[i,] = (imread('./Dataset/Patches_Pre_64/'+ self.partition + '/Images/' + ID_path))[:,:,:,:2]

            mask[i,] = imread('./Dataset/Patches_Pre_64/'+ self.partition + '/Masks/'+ ID_path)

        return X, mask

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def order_dirs(self, element):
        a = element.split("_")[2]
        return int(a.split(".")[0])
