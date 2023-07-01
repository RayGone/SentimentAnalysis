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
from tensorflow.keras.layers import Flatten,Embedding,Dense,Dropout,Conv1D,MaxPool1D
import gc

def seed_everything(seed=0):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(rand_seed)

rand_seed = 999
seed_everything(rand_seed)

use_pre_trained_embd_layer = False
use_googletrans_aug_data = False
save_model = False

def preTrainEmbedding(embeddinglayer,train=[],val=[]):
    model = Sequential([
        embeddinglayer,
        Dropout(0.1),
        Flatten(),
        Dense(64,activation='relu'),
        Dropout(0.3),
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
    history = model.fit(tf.constant(train[0]),
                          tf.constant(train[1]),
                          epochs=5,verbose=1,
                          validation_data=val
                      )
    
    print(history.history)
    return embeddinglayer

def LabelEncoding(x):
    if x==0:
        return [1,0,0]
    if x==1:
        return [0,1,0]
    if x==-1:
        return [0,0,1]
    
    return x
 
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

# if include_MLP_new_neutral_aug:
#     print("\n\n*******Using Aggregated Data*********\n")
#     print("Here we add data in neutral class which are collected from news portals and are labeled as neutral by MLP model. (MLP_4_Sentiment_Analysis.py)")
#     agg_data = datasets.Dataset.from_pandas(pd.read_csv("data_dump/aggregated.csv")[['text']])
#     agg_data = agg_data.add_column('Sentiment',np.zeros(agg_data.num_rows,dtype='int32'))
#     agg_data = agg_data.rename_column(original_column_name='text',new_column_name='Sentences')
    
#     nepCov19['train'] = datasets.concatenate_datasets([nepCov19['train'],agg_data])

max_len = 95
# tokenizer = PreTrainedTokenizerFast.from_pretrained("raygx/GPT2-Nepali-Casual-LM") ### 50,000 tokens
tokenizer = PreTrainedTokenizerFast.from_pretrained("raygx/Covid-News-Headline-Generator") ### 30,000 tokens

nepCov19 = nepCov19['train'].train_test_split(test_size=0.2)
print("Dataset",nepCov19)

train_input = tokenizer(
                    nepCov19['train']['Sentences']
                ).input_ids

test_input = tokenizer(
                nepCov19['test']['Sentences']
                ).input_ids

# max_len = max([max([len(x) for x in tokenizer(nepCov19['train']['Sentences']).input_ids]),max([len(x) for x in tokenizer(nepCov19['test']['Sentences']).input_ids])]) + 10

train_input = pad_sequences(
                        train_input,
                        maxlen = max_len,
                        padding='post',
                        value=tokenizer.pad_token_id
                    )
train_labels = [LabelEncoding(x) for x in nepCov19['train']['Sentiment']]

test_input = pad_sequences(
                        test_input,
                        maxlen = max_len,
                        padding='post',
                        value=tokenizer.pad_token_id
                    )
test_labels = [LabelEncoding(x) for x in nepCov19['test']['Sentiment']]

print("All True Labels",tf.math.confusion_matrix([np.argmax(x) for x in train_labels+test_labels],[np.argmax(x) for x in train_labels+test_labels],num_classes=3))

embd_layer = Embedding(len(tokenizer), 380, input_length=max_len)
if use_pre_trained_embd_layer:
    print("\n****Using Pre-Trained Embedding Layer****")
    embd_layer = tf.keras.models.load_model("saved_models/MLP_4_SA").get_layer(index=0)
else:
    print("*** Pre-Training a Embedding Layer ****")
    embd_layer = preTrainEmbedding(embd_layer,
                                train=[tf.constant(train_input),tf.constant(train_labels)],
                                val=[tf.constant(test_input),tf.constant(test_labels)])
    
try:
    raise("Let's Build New Model")
    print("Loading saved model")
    model = tf.keras.models.load_model("saved_models/Conv_4_SA")
    print(model.summary())
except:
    model = Sequential()
    model.add(embd_layer)
    model.add(Conv1D(128,5,activation='relu'))
    model.add(Dropout(0.3))
    model.add(MaxPool1D(3))
    model.add(Conv1D(64,3,activation='relu'))
    model.add(Dropout(0.3))
    model.add(MaxPool1D(2))
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.3))
    # model.add(Dense(128,activation='relu'))
    # model.add(Dropout(0.2))
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
        model.save(os.path.join(os.getcwd(),"saved_models/Conv_4_SA"))


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
"""
# ***********BEST_RESULT***************************** 
# =================================================================
Total params: 21,235,587
Trainable params: 21,235,587
Non-trainable params: 0
# _________________________________________________________________

******Evaluations***********

260/260 [==============================] - 1s 3ms/step
F1-Score 0.8364888987329185
Precision-Score 0.8369815220058648
Recall-Score 0.8364620938628159
Accuracy-Score 0.8364620938628159
tf.Tensor(
[[2190  235  158]
 [ 185 2520  264]
 [ 173  344 2241]], shape=(3, 3), dtype=int32)
True Labels Onlys tf.Tensor(
[[2583    0    0]
 [   0 2969    0]
 [   0    0 2758]], shape=(3, 3), dtype=int32)
"""