import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import datasets
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Embedding,Dense,Dropout,Softmax
import gc

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(rand_seed)

use_googletrans_aug_data = False
rand_seed = 99
seed_everything(rand_seed)


def preTrainEmbedding(embeddinglayer,data,label):
    model = Sequential([
        embeddinglayer,
        Dropout(0.1),
        Flatten(),
        # Dense(512,activation='relu'),
        # Dropout(0.4),
        Dense(3,activation='sigmoid')
    ])
    
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
    history = model.fit(tf.constant(data),
                tf.constant(label),
                epochs=1
            )
    
    print(history.history)
    return embeddinglayer

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
    
nepCov19 = load_dataset("raygx/NepCov19TweetsPlus").shuffle(rand_seed)

if use_googletrans_aug_data:
    print("\n\nAdding Data to Neutral class \n- augmented through googletrans \n- ne-2-en-2-ne")
    aug_data = pd.read_csv("augment/googletrans_augmented_data.csv")
    aug_data = aug_data.rename(columns={"Unnamed: 0":"Sentiment","ne":"Sentences"})
    aug_data['Sentiment'] = np.zeros(aug_data.shape[0],dtype=np.int32)
    nepCov19 = pd.concat([nepCov19['train'].to_pandas(),aug_data]).drop_duplicates()
    # print(nepCov19['Sentiment'].value_counts())
    nepCov19 = datasets.DatasetDict({
        'train':datasets.Dataset.from_pandas(nepCov19)   
        })  
    
    
print(nepCov19)

max_len = 95
tokenizer = PreTrainedTokenizerFast.from_pretrained("raygx/GPT2-Nepali-Casual-LM")
print("Vocab Size",len(tokenizer))

nepCov19 = nepCov19['train'].train_test_split(test_size=0.2)
print("Dataset",nepCov19)
train_input = pad_sequences(
                        tokenizer(
                            nepCov19['train']['Sentences']
                            ).input_ids,
                        maxlen = max_len,
                        padding='post',
                        value=tokenizer.pad_token_id
                    )
train_labels = [LabelEncoding(x) for x in nepCov19['train']['Sentiment']]

test_input = pad_sequences(
                        tokenizer(
                            nepCov19['test']['Sentences']
                            ).input_ids,
                        maxlen = max_len,
                        padding='post',
                        value=tokenizer.pad_token_id
                    )
test_labels = [LabelEncoding(x) for x in nepCov19['test']['Sentiment']]

try:
    raise("Let's Build New Model")
    print("Loading saved model")
    model = tf.keras.models.load_model("saved_models/MLP_4_SA")
    print(model.summary())
except:
    model = Sequential()
    embd_layer = Embedding(len(tokenizer), 380, input_length=max_len)
    
    print("*** Pre-Training a Embedding Layer ****")
    embd_layer = preTrainEmbedding(embd_layer,data=np.concatenate([train_input,test_input]),label=np.concatenate([train_labels,test_labels]))
    
    model.add(embd_layer)
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3,activation='sigmoid'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                    initial_learning_rate=0.00001,
                    decay_steps=100000,                
                    decay_rate=0.95,
                    staircase=True
                )
            ),
        loss='categorical_crossentropy',
        metrics=['acc'])
    
    print(model.summary())
    
    history = model.fit(tf.constant(train_input),
            tf.constant(train_labels),
            epochs=30,
            validation_data=[tf.constant(test_input),tf.constant(test_labels)],
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

    260/260 [==============================] - 1s 2ms/step
    F1-Score 0.811957794862897
    Precision-Score 0.8124831706395869
    Recall-Score 0.811913357400722
    Accuracy-Score 0.811913357400722
    tf.Tensor(
    [[2136  265  231]
    [ 228 2504  284]
    [ 174  381 2107]], shape=(3, 3), dtype=int32)
    True Labels Onlys tf.Tensor(
    [[2632    0    0]
    [   0 3016    0]
    [   0    0 2662]], shape=(3, 3), dtype=int32)
"""