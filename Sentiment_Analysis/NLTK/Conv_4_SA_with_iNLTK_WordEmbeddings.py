### USING iNLTK sentence embeddings

import os
import random
import numpy as np
import pandas as pd
import datasets

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense,Dropout,Conv1D,MaxPool1D

from package.DataGenerator import DataGenerator

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    # tf.keras.utils.set_random_seed(rand_seed)
    
rand_seed = 99
seed_everything(rand_seed)
    
data = datasets.load_dataset("raygx/NepCov19TweetsPlus")
data = data.rename_column(original_column_name='Sentences',new_column_name='text')
data = data.rename_column(original_column_name='Sentiment',new_column_name='label')
data = data['train'].train_test_split(test_size=0.2) 

train_gen = DataGenerator(data['train'],max_token_len=110)
test_gen= DataGenerator(data['test'],max_token_len=110)


model = Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(110,400)))
model.add(Conv1D(32,5,activation='relu'))
model.add(Dropout(0.3))
model.add(MaxPool1D(3))
model.add(Flatten())
model.add(Dense(512,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(512,activation='relu'))
# model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3,activation='sigmoid'))

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.0001,
                decay_steps=100000,                
                decay_rate=0.95,
                staircase=True
            )
        ),
    loss='categorical_crossentropy',
    metrics=['acc'])

print(model.summary())

history = model.fit(train_gen,
        epochs=100,
        validation_data=test_gen,
        callbacks=[tf.keras.callbacks.EarlyStopping(
                            monitor='val_acc', patience=3,
                            verbose=1, mode='max',
                            restore_best_weights=True)
                        ])

# model.save("saved_models/MLP_4_SA")
    
####-----------------------------------------
## ---------------Something------------------
####-----------------------------------------
print("l\n\n******Evaluations***********\n")

pred_labels = [np.argmax(x) for x in 
        tf.nn.softmax(
            model.predict(
                x=tf.constant(test_input)
            )
        )
    ]

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
test_labels = [np.argmax(x) for x in test_labels]
print("F1-Score",f1_score(test_labels,pred_labels,average='weighted'))
print("Precision-Score",precision_score(test_labels,pred_labels,average='weighted'))
print("Recall-Score",recall_score(test_labels,pred_labels,average='weighted'))
print("Accuracy-Score",accuracy_score(test_labels,pred_labels))

from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

confusion_matrix = tf.math.confusion_matrix(test_labels,pred_labels,num_classes=3)
print(confusion_matrix)
cmd = ConfusionMatrixDisplay(confusion_matrix.numpy())
cmd.plot()
# plt.show()

print("True Labels Onlys",tf.math.confusion_matrix(test_labels,test_labels,num_classes=3))
# print("True Labels Onlys",tf.math.confusion_matrix(labels,labels,num_classes=3))

"""
    RESULT:  
    =================================================================
    Total params: 28,275,139
    Trainable params: 28,275,139
    Non-trainable params: 0
    _________________________________________________________________
    Epoch 18/30
    1039/1039 [==============================] - 22s 21ms/step - loss: 0.1402 - acc: 0.9605 - val_loss: 0.5335 - val_acc: 0.8119

    ******Evaluations***********
    
    F1-Score 0.7618125631946366
    Precision-Score 0.7624203603192438
    Recall-Score 0.7616125150421179
    Accuracy-Score 0.7616125150421179
    tf.Tensor(
    [[2026  333  273]
    [ 270 2327  419]
    [ 223  463 1976]], shape=(3, 3), dtype=int32)
    True Labels Onlys tf.Tensor(
    [[2632    0    0]
    [   0 3016    0]
    [   0    0 2662]], shape=(3, 3), dtype=int32)
"""