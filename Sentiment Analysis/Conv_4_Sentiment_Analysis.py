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
    tf.keras.utils.set_random_seed(rand_seed)

rand_seed = 9
seed_everything(rand_seed)

use_googletrans_aug_data = True
# include_MLP_new_neutral_aug = False
save_model = True

def LabelEncoding(x):
    if x==0:
        return [1,0,0]
    if x==1:
        return [0,1,0]
    if x==-1:
        return [0,0,1]
    
    return x
 
nepCov19 = load_dataset("raygx/NepCov19Tweets").shuffle(rand_seed)

if use_googletrans_aug_data:
    print("\n\nAdding Data to Neutral class \n- augmented through googletrans \n- ne-2-en-2-ne")
    aug_data = pd.read_csv("augment/googletrans_augmented_data.csv")
    aug_data = aug_data.rename(columns={"Unnamed: 0":"Sentiment","ne":"Sentences"})
    aug_data['Sentiment'] = np.zeros(aug_data.shape[0],dtype=np.int32)
    nepCov19 = datasets.DatasetDict({
        'train':datasets.concatenate_datasets([
                    nepCov19.filter(lambda x: x['Sentiment']!=0)['train'], # because augdata already contains original as well
                    datasets.Dataset.from_pandas(aug_data) 
                ])     
        })  

# if include_MLP_new_neutral_aug:
#     print("\n\n*******Using Aggregated Data*********\n")
#     print("Here we add data in neutral class which are collected from news portals and are labeled as neutral by MLP model. (MLP_4_Sentiment_Analysis.py)")
#     agg_data = datasets.Dataset.from_pandas(pd.read_csv("data_dump/aggregated.csv")[['text']])
#     agg_data = agg_data.add_column('Sentiment',np.zeros(agg_data.num_rows,dtype='int32'))
#     agg_data = agg_data.rename_column(original_column_name='text',new_column_name='Sentences')
    
#     nepCov19['train'] = datasets.concatenate_datasets([nepCov19['train'],agg_data])

max_len = 95
tokenizer = PreTrainedTokenizerFast.from_pretrained("raygx/GPT2-Nepali-Casual-LM")

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

print("All True Labels",tf.math.confusion_matrix([np.argmax(x) for x in train_labels+test_labels],[np.argmax(x) for x in train_labels+test_labels],num_classes=3))

embd_layer = Embedding(len(tokenizer), 380, input_length=max_len)
try:
    raise("Let's Build New Model")
    print("Loading saved model")
    model = tf.keras.models.load_model("saved_models/MLP_4_SA")
    print(model.summary())
except:
    model = Sequential()
    model.add(embd_layer)
    model.add(Conv1D(64,5,activation='relu'))
    model.add(MaxPool1D(3))
    model.add(Conv1D(32,5,activation='relu'))
    model.add(MaxPool1D(3))
    model.add(Flatten())
    model.add(Dense(512,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3,activation='sigmoid'))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00008),
        loss='categorical_crossentropy',
        metrics=['acc',tf.keras.metrics.Precision()])

    print(model.summary())

    history = model.fit(tf.constant(train_input),
            tf.constant(train_labels),
            epochs=5)

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
-- Using NepCov19Tweets Dataset As it is --
    BEST_RESULT: 
        #### No Aug Data
        F1-Score: 0.7022021185799427
        Precision-Score: 0.700344004267401
        Recall-Score: 0.7054518297236744
        accuracy_Score: 0.7054518297236744
        confusion matrix:
                [[ 366  358  224]
                [ 236 2304  432]
                [ 217  505 2053]]
    HyperParameters:        
        rand_seed: 9 ## Seed for model weights and train_test data shuffle
        epochs: 5
        max_len: 95 ## maximum input length
        embedding_size: 380
        optimizer: tf.keras.optimizers.Adam(lr=0.000099)
        loss: sparse_categorical_crossentropy
        Conv1_l1: 
            units: (32,5)
            activation: relu
        maxpool_l1: 3
        Dense_1: 
            units: 100 
            activation: relu
        Dense_2: 
            units: 3 
            activation: sigmoid
"""