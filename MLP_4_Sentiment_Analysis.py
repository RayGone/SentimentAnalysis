import numpy as np
import tensorflow as tf
from datasets import load_dataset
from transformers import PreTrainedTokenizerFast

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Embedding,Dense
import gc

def OneHotEncoding(x):
    if x==-1:
        return 2
        return [1,0,0]
    if x==0:
        return 0
        return [0,1,0]
    if x==1:
        return 1
        return [0,0,1]
    
nepCov19 = load_dataset("raygx/NepCov19Tweets").shuffle(999)
data_len = nepCov19['train'].num_rows
print(nepCov19)

labels = [OneHotEncoding(x) for x in nepCov19['train']['Sentiment']]

tokenizer = PreTrainedTokenizerFast.from_pretrained("raygx/GPT2-Nepali-Casual-LM")
print("Vocab Size",len(tokenizer))
nepCov19 = tokenizer(nepCov19['train']['Sentences']).input_ids

max_len = 100
input = pad_sequences(nepCov19,maxlen = max_len,padding='post',value=tokenizer.pad_token_id)

model = Sequential()
embd_layer = Embedding(len(tokenizer), 128, input_length=max_len)
model.add(embd_layer)
model.add(Flatten())
model.add(tf.keras.layers.Conv1D(128,5,activation='gelu'))
model.add(tf.keras.layers.MaxPool1D(3))
model.add(Dense(128,activation='gelu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(3,activation=tf.nn.softmax))
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['acc'])
print(model.summary())

train_block = int((data_len*0.1)/10)
print("Training Size",train_block)

history = model.fit(tf.constant(input[:train_block]),
          tf.constant(labels[:train_block]),
          epochs=10,validation_split=0.2)

eval = model.evaluate(
    x=tf.constant(input[train_block:]),
    y=tf.constant(labels[train_block:]),
    return_dict=True
)

print(history)
print(eval)
