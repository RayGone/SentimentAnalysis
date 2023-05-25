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

use_googletrans_aug_data = True
rand_seed = 9
seed_everything(rand_seed)

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
    
nepCov19 = load_dataset("raygx/NepCov19Tweets").shuffle(rand_seed)

if use_googletrans_aug_data:
    print("\n\nAdding Data to Neutral class \n- augmented through googletrans \n- ne-2-en-2-ne")
    aug_data = pd.read_csv("augment/googletrans_augmented_data.csv")
    aug_data = aug_data.rename(columns={"Unnamed: 0":"Sentiment","ne":"Sentences"})
    aug_data['Sentiment'] = np.zeros(aug_data.shape[0],dtype=np.int32)
    nepCov19 = datasets.DatasetDict({
        'train':datasets.concatenate_datasets([
                    nepCov19.filter(lambda x: x['Sentiment']!=0)['train'],
                    datasets.Dataset.from_pandas(aug_data) 
                ])     
        })  
    
    # print(nepCov19['train'].to_pandas()['Sentiment'].value_counts())
    
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
    model.add(embd_layer)
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dropout(0.4))
    model.add(Dense(128,activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3,activation='sigmoid'))
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
        loss='categorical_crossentropy',
        metrics=['acc',tf.keras.metrics.Precision()])

    print(model.summary())


    print("Training The Model")
    history = model.fit(tf.constant(train_input),
            tf.constant(train_labels),
            epochs=5)#, validation_split=0.1)

    model.save("saved_models/MLP_4_SA")
    
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
        #### No Aug Data
        F1-Score 0.6868384368625976
        Precision-Score 0.6839599853659085
        Recall-Score 0.7035100821508589
        Accuracy-Score 0.7035100821508589
        confusion matrix:
            [[ 207  417  325]
            [ 134 2346  448]
            [ 122  539 2157]]
        Epoch 5/5
        837/837 [==============================] - 19s 23ms/step - loss: 0.4255 - acc: 0.8412 - precision: 0.7573
        
        #### With Aug data
        F1-Score 0.7364478486255656
        Precision-Score 0.7374669280946146
        Recall-Score 0.7364838329624296
        Accuracy-Score 0.7364838329624296
        confusion matrix:
            [[1418  273  224]
             [ 314 2255  393]
             [ 281  528 1953]]
        Epoch 5/5
        955/955 [==============================] - 22s 23ms/step - loss: 0.2952 - acc: 0.9049 - precision: 0.7841
    HyperParameters:        
        rand_seed = 9 ## Seed for model weights and train_test data shuffle
        epochs = 5
        max_len = 95 ## maximum input length
        embedding_size = 380
        optimizer = tf.keras.optimizers.Adam(lr=0.000099)
        metrics = ['acc','precision']
        loss = sparse_categorical_crossentropy
        Dense_1 = 256, activation = relu
        dropout = 0.4
        Dense_2 = 128, activation = relu
        dropout = 0.2
        Dense_3 = 3, activation = sigmoid
"""