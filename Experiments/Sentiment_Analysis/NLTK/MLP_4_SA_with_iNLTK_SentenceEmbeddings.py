### USING iNLTK sentence embeddings

import os
import random
import numpy as np
import datasets

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Embedding,Dense,Dropout,Softmax

from package.Embeddings import getSentenceEmbeddings

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(rand_seed)
    tf.random.set_seed(seed) # tensorflow
    
rand_seed = 99
seed_everything(rand_seed)

### -----------\\//------------ ###

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
    
if os.path.exists("Sentiment_Analysis/NLTK/sentence_embeddings"):
    print("loading from disk")
    data = datasets.Dataset.load_from_disk("Sentiment_Analysis/NLTK/sentence_embeddings")
else:   
    print("getSentenceEmbeddings()")
    data = getSentenceEmbeddings()
    data = datasets.Dataset.from_pandas(data)#.shuffle(rand_seed)
    data.save_to_disk("NLTK/sentence_embeddings")

data = data.shuffle(rand_seed).train_test_split(test_size=0.2)
print(data)

max_len = len(data['train'][0]['sent_embd'])

train_labels = [LabelEncoding(x) for x in data['train']['label']]
test_labels = [LabelEncoding(x) for x in data['test']['label']]

model = Sequential()
model.add(tf.keras.layers.InputLayer(input_shape=(max_len,)))
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256,activation='relu'))
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(3,activation='sigmoid'))

model.compile(
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=0.001,
                decay_steps=10000,                
                decay_rate=0.95,
                staircase=True
            )
        ),
    loss='categorical_crossentropy',
    metrics=['acc'])

print(model.summary())

history = model.fit(tf.constant(data['train']['sent_embd']),
        tf.constant(train_labels),
        epochs=100,
        validation_data=(tf.constant(data['test']['sent_embd']),tf.constant(test_labels)),
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
                x=tf.constant(data['test']['sent_embd'])
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
Total params: 706,307
Trainable params: 706,307
Non-trainable params: 0
_________________________________________________________________
Epoch 5/100
1039/1039 [==============================] - 3s 3ms/step - loss: 0.5436 - acc: 0.7706 - val_loss: 0.7814 - val_acc: 0.6732

******Evaluations***********

F1-Score 0.6729567546716254
Precision-Score 0.6791754694996459
Recall-Score 0.6731648616125151
Accuracy-Score 0.6731648616125151
tf.Tensor(
[[1700  630  320]
 [ 312 2219  457]
 [ 299  698 1675]], shape=(3, 3), dtype=int32)
"""