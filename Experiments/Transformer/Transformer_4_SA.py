import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import datasets
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

from tensorflow.keras.preprocessing.sequence import pad_sequences

from EncoderTransformer import Transformer

import gc

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
    
nepCov19 = load_dataset("raygx/NepCov19TweetsPlus").shuffle(rand_seed)  

max_len = 128
tokenizer = PreTrainedTokenizerFast.from_pretrained("raygx/BERT_Nepali_Tokenizer")
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

gc.collect()

model = Transformer(num_layers=2,d_model=256,GSA_num_heads=8,
                    LSA_num_heads=2,LSA_num_window=4,dff=512,
                    vocab_size=len(tokenizer),num_class=3)

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


dummy = np.ones((32,max_len))
model(dummy) ## building the model
print(model.summary())

history = model.fit(tf.constant(train_input),
        tf.constant(train_labels),
        epochs=30,batch_size=32,
        validation_data=[tf.constant(test_input),tf.constant(test_labels)],
        callbacks=[tf.keras.callbacks.EarlyStopping(
                            monitor='val_acc', patience=3,
                            verbose=1, mode='max',
                            restore_best_weights=True)
                        ])
    
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
    Experiment: Transformer(num_layers=1,d_model=256,GSA_num_heads=8,
                    LSA_num_heads=2,LSA_num_window=4,dff=512,
                    vocab_size=len(tokenizer),num_class=3)
                    
    Result: 
        Epoch 11/30
        1039/1039 [==============================] - 87s 83ms/step - loss: 0.3976 - acc: 0.8535 - val_loss: 0.6484 - val_acc: 0.7557
        
        F1-Score 0.7559914005465729
        Precision-Score 0.7620842599277837
        Recall-Score 0.7556866048862679
        Accuracy-Score 0.7556866048862679
        tf.Tensor(
        [[1808  457  290]
        [ 179 2422  398]
        [ 167  539 2049]], shape=(3, 3), dtype=int32)
        True Labels Onlys tf.Tensor(
        [[2555    0    0]
        [   0 2999    0]
        [   0    0 2755]], shape=(3, 3), dtype=int32)
"""