import tensorflow as tf

import numpy as np
import pandas as pd
from datasets import load_dataset

from package.Embeddings import getWordEmbeddings

def LabelEncoding(x):
    if x == -1:
        # return 2
        x = [0,0,1]
    elif x == 0:
        # return 0
        x = [1,0,0]
    elif x == 1:
        # return 1
        x = [0,1,0]
    
    return np.array(x)
    
class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, data, max_token_len=110, batch_size=32,shuffle=True):
        'Initialization'
        self.max_len = max_token_len
        self.batch_size = batch_size
        self.data = data
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(self.data.num_rows / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        X, y = self.__data_generation(indexes)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(self.data.num_rows)
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        chunk = self.data.select(indexes)
        X = getWordEmbeddings(chunk['text'],max_len = self.max_len)
        y = np.array([LabelEncoding(i) for i in chunk['label']])
        return X,y


if __name__ == "__main__":
    # print("Hello World!!")
    data = load_dataset("raygx/NepCov19Tweets")
    data = data.rename_column(original_column_name='Sentences',new_column_name='text')
    data = data.rename_column(original_column_name='Sentiment',new_column_name='label')
    data = data['train'].train_test_split(test_size=0.2)
    
    train_data_gen = DataGenerator(data['train'])
    
    for batch in train_data_gen:
        print(batch)
        break