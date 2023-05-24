"""
    The purpose of this script is to identify probable texts with "neutral" sentiment
    from the corpus of texts crawled from news sites (in folder ./data_crawler/)
    
    The Model is trained in MLP_4_Sentiment_Analysis.py
    The Model is stored in saved_models/MLP_4_SA
    
    The output is saved in data_dump/MLP_4_SA_neutral_labeled_news.csv
"""

import pandas as pd
import numpy as np
import tensorflow as tf
from transformers import PreTrainedTokenizerFast
from tensorflow.keras.preprocessing.sequence import pad_sequences

csv_files = [f for f in os.listdir("data_crawler") if '.csv' in f and 'clean' in f]

data = pd.concat([
                    pd.read_csv("data_crawler/{}".format(cf),engine='python') for cf in csv_files
                ])

data.drop(["Unnamed: 0"],axis=1,inplace=True)
data.drop_duplicates(inplace=True)
data.reset_index(inplace=True)
print(data.head())


tokenizer = PreTrainedTokenizerFast.from_pretrained("raygx/GPT2-Nepali-Casual-LM")
print("Vocab Size",len(tokenizer))
input_ids = tokenizer(data['text'].to_list()).input_ids

max_len = 95 ## model max_len
input_ids = pad_sequences(input_ids,maxlen = max_len,padding='post',value=tokenizer.pad_token_id)

model = tf.keras.models.load_model("saved_models/Conv_4_SA")

pred_labels = [np.argmax(x) for x in 
        tf.nn.softmax(
            model.predict(
                x=tf.constant(input_ids)
            )
        )
    ]

label_index = [x for x in range(data.shape[0]) if pred_labels[x] == 0]
result = data.loc[label_index]
result.index = range(result.shape[0])
result.drop(['index'],axis=1,inplace=True)
result.to_csv("data_dump/Conv_4_SA_neutral_labeled_news.csv")


# import seaborn
# from matplotlib import pyplot as plt

# seaborn.histplot(pred_labels)
# plt.show()
