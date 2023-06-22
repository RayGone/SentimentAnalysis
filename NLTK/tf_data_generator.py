import tensorflow as tf
import tensorflow_datasets as tfds

import numpy as np
import pandas as pd
from datasets import load_dataset

from inltk.inltk import setup,tokenize,get_embedding_vectors, get_sentence_encoding
setup('ne')

def LabelEncoding(x):
    if x == -1:
        # return 2
        return [0,0,1]
    if x == 0:
        # return 0
        return [1,0,0]
    if x == 1:
        # return 1
        return [0,1,0]
    
class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, labels, max_seq_len=100, batch_size=32,
                 n_classes=3, shuffle=True):
        'Initialization'
        self.max_len = max_seq_len
        self.batch_size = batch_size
        self.labels = labels
        self.text = data
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.embd = [0 for i in range(len(labels))]
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(len(self.labels) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.labels))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = []
        y = []
        for i in indexes:
            if not self.embd[i]:
                tX = get_embedding_vectors(self.text[i],'ne')
                iL = len(tX)
                jL = len(tX[0])

                tmp = np.zeros((self.max_len,jL))
                tmp -= 0.00001
                if(iL>self.max_len):
                    print("Max input_len",iL)
                    tmp[:,:] = tX[:self.max_len]
                    tX = tmp
                if(iL<self.max_len):
                    tmp[:iL,:] = tX
                    tX = tmp
                    
                self.embd[i] = tX
                
            X.append(self.embd[i])
            y.append(tf.constant(LabelEncoding(self.labels[i])))
        return X,y
