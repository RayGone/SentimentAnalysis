#!/usr/bin/env python
# coding: utf-8


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
from tensorflow.keras.layers import Flatten,Embedding,Dense,Bidirectional,GRU,Dropout
import gc

rand_seed = 999
use_pre_trained_embd_layer = False
use_googletrans_aug_data = False
save_model = False

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(rand_seed)
    tf.random.set_seed(seed) # tensorflow
    
seed_everything(rand_seed)


nepCov19 = load_dataset("raygx/NepCov19TweetsPlus").shuffle(rand_seed)

if use_googletrans_aug_data:
    print("\nAdding Data to Neutral class \n- augmented through googletrans \n- ne-2-en-2-ne\n")
    aug_data = pd.read_csv("augment/googletrans_augmented_data.csv")
    aug_data = aug_data.rename(columns={"Unnamed: 0":"Sentiment","ne":"Sentences"})
    aug_data['Sentiment'] = np.zeros(aug_data.shape[0],dtype=np.int32)
    nepCov19 = datasets.DatasetDict({
        'train':datasets.concatenate_datasets([
                    nepCov19.filter(lambda x: x['Sentiment']!=0)['train'], # because augdata already contains original as well
                    datasets.Dataset.from_pandas(aug_data) 
                ])     
        })  
    
nepCov19

# tokenizer = PreTrainedTokenizerFast.from_pretrained("raygx/GPT2-Nepali-Casual-LM") ### 50,000 tokens
tokenizer = PreTrainedTokenizerFast.from_pretrained("raygx/Covid-News-Headline-Generator") ### 30,000 tokens
max_len = 95

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
                epochs=1,verbose=2
            )
    
    print(history.history)
    return embeddinglayer

### -----------\\//------------ ###

def LabelEncoding(x):
    if x==0:
        return [1,0,0]
    if x==1:
        return [0,1,0]
    if x==-1:
        return [0,0,1]
    
    return x
### ----------\\//---------- ###

nepCov19 = nepCov19['train'].train_test_split(test_size=0.2)
print("Dataset",nepCov19)

print("Preparing Training Input and Labels")
train_input = pad_sequences(
                        tokenizer(
                            nepCov19['train']['Sentences']
                            ).input_ids,
                        maxlen = max_len,
                        padding='post',
                        value=tokenizer.pad_token_id
                    )
train_labels = [LabelEncoding(x) for x in nepCov19['train']['Sentiment']]

print("Preparing Test Input and Labels")
test_input = pad_sequences(
                        tokenizer(
                            nepCov19['test']['Sentences']
                            ).input_ids,
                        maxlen = max_len,
                        padding='post',
                        value=tokenizer.pad_token_id
                    )
test_labels = [LabelEncoding(x) for x in nepCov19['test']['Sentiment']]

# cnf = tf.math.confusion_matrix(
#                 [np.argmax(x) for x in train_labels+test_labels],[np.argmax(x) for x in train_labels+test_labels],num_classes=3
#             ).numpy()
# print(cnf)


### https://stats.stackexchange.com/questions/181/
#### /how-to-choose-the-number-of-hidden-layers-and-nodes-in-a-feedforward-neural-netw/136542#136542
n_hidden = int(len(train_labels)/(2*(95 + 3)))


try:
    raise("Let's Build New Model")
    print("Loading saved model")
    model = tf.keras.models.load_model("saved_models/LSTM_4_SA")
    print(model.summary())
except:
    embd_layer = Embedding(len(tokenizer), 380, input_length=max_len)
    
    if use_pre_trained_embd_layer:
        print("\n****Using Pre-Trained Embedding Layer****\n")
        embd_layer = tf.keras.models.load_model("saved_models/Conv_4_SA").get_layer(index=0)
    else:
        print("*** Pre-Training a Embedding Layer ****")
        embd_layer = preTrainEmbedding(embd_layer,data=np.concatenate([train_input,test_input]),label=np.concatenate([train_labels,test_labels]))
        
    model = Sequential()
    model.add(embd_layer)
    # with tf.device('/device:CPU:0'):#For GRU - CPU works better
    model.add(Bidirectional(GRU(n_hidden*2,return_sequences=True)))
    model.add(Dropout(0.3))
    model.add(Bidirectional(GRU(n_hidden*2)))
    model.add(Dropout(0.3))
    model.add(Flatten())
        
    model.add(Dense(n_hidden,activation='relu'))
    model.add(Dropout(0.25))
    model.add(Dense(n_hidden,activation='relu'))
    model.add(Dropout(0.25))
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
    
    if save_model:
        print("Saving the model")
        model.save(os.path.join(os.getcwd(),"saved_models/LSTM_4_SA"))
        

print("\n\n******Evaluations***********\n")
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
print("accuracy_Score",accuracy_score(test_labels,pred_labels))


from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

confusion_matrix = tf.math.confusion_matrix(test_labels,pred_labels,num_classes=3)
print(confusion_matrix)
cmd = ConfusionMatrixDisplay(confusion_matrix.numpy(),
            display_labels=['Neutral',"Positive","Negative"])
cmd.plot()
# plt.show()
print("True Labels Onlys",tf.math.confusion_matrix(test_labels,test_labels,num_classes=3))

'''
    *********** Best Result ***************

******Evaluations***********

260/260 [==============================] - 7s 23ms/step
F1-Score 0.7799774640346219
Precision-Score 0.782310455826575
Recall-Score 0.7795161872668191
accuracy_Score 0.7795161872668191
tf.Tensor(
[[2045  335  297]
 [ 214 2342  377]
 [ 165  444 2090]], shape=(3, 3), dtype=int32)
True Labels Onlys tf.Tensor(
[[2677    0    0]
 [   0 2933    0]
 [   0    0 2699]], shape=(3, 3), dtype=int32)
'''