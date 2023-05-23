import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Embedding,Dense
import gc

rand_seed = 99
tf.keras.utils.set_random_seed(rand_seed)

def OneHotEncoding(x):
    if x == -1:
        return 2
        return [1,0,0]
    if x == 0:
        return 0
        return [0,1,0]
    if x == 1:
        return 1
        return [0,0,1]
    
nepCov19 = load_dataset("raygx/NepCov19Tweets").shuffle(rand_seed)
data_len = nepCov19['train'].num_rows
print(nepCov19)

labels = [OneHotEncoding(x) for x in nepCov19['train']['Sentiment']]

tokenizer = PreTrainedTokenizerFast.from_pretrained("raygx/GPT2-Nepali-Casual-LM")
print("Vocab Size",len(tokenizer))
nepCov19 = tokenizer(nepCov19['train']['Sentences']).input_ids

max_len = 95
input = pad_sequences(nepCov19,maxlen = max_len,padding='post',value=tokenizer.pad_token_id)

train_block = int((data_len*8)/10)
print("Training Size",train_block)

try:
    raise("Let's Build New Model")
    print("Loading saved model")
    model = tf.keras.models.load_model("saved_models/MLP_4_SA")
    print(model.summary())
except:
    model = Sequential()
    embd_layer = Embedding(len(tokenizer), 480, input_length=max_len)
    model.add(embd_layer)
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dense(128,activation='sigmoid'))
    model.add(Dense(3,activation=tf.nn.softmax))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(lr=0.00005),
        loss='sparse_categorical_crossentropy',
        metrics=['acc'])

    print(model.summary())


    print("Training The Model")
    history = model.fit(tf.constant(input[:train_block]),
            tf.constant(labels[:train_block]),
            epochs=5, validation_split=0.1)

    model.save("saved_models/MLP_4_SA")
    
####-----------------------------------------
## ---------------Something------------------
####-----------------------------------------
print("l\n\n******Evaluations***********\n")
pred_labels = [np.argmax(x) for x in 
        tf.nn.softmax(
            model.predict(
                x=tf.constant(input[train_block:])
            )
        )
    ]

from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

print("F1-Score",f1_score(labels[train_block:],pred_labels,average='weighted'))
print("Precision-Score",precision_score(labels[train_block:],pred_labels,average='weighted'))
print("Recall-Score",recall_score(labels[train_block:],pred_labels,average='weighted'))
print("accuracy_Score",accuracy_score(labels[train_block:],pred_labels))


from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

confusion_matrix = tf.math.confusion_matrix(labels[train_block:],pred_labels,num_classes=3)
print(confusion_matrix)
cmd = ConfusionMatrixDisplay(confusion_matrix.numpy())
cmd.plot()
# plt.show()