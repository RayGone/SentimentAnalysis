import pandas as pd
import numpy as np
from datasets import load_dataset

from inltk.inltk import setup,tokenize,get_embedding_vectors, get_sentence_encoding
setup('ne')


def getWordEmbeddings():
    data = load_dataset('raygx/NepCov19TweetsPlus')
    print(data)

    labels = data['train']['Sentiment']
    print("Getting Word Embeddings")
    word_embd = get_embedding_vectors(data['train']['Sentences'],'ne')
    max_word_embd_len = max([len(x) for x in word_embd])
    print("Max Sentence Length:",max_word_embd_len)

    print("Adding padding vectors to embedding vectors")
    for i in range(len(word_embd)):
        tmp = np.zeros((max_word_embd_len,400)) 
        tmp -= 0.00001
        tmp[:word_embd[i].shape[0],:] = word_embd[i]
        word_embd[i] = tmp
        print(tmp.shape)

    pdf = pd.DataFrame({
            "label": labels,
            "word_embd": word_embd
        })
    return pdf

def getSentenceEmbeddings():  
    print("Loading Data")  
    data = load_dataset('raygx/NepCov19TweetsPlus')
    print(data)

    labels = data['train']['Sentiment']
    print("Getting Sentence Embeddings")
    sent_embd = [get_sentence_encoding(x,'ne') for x in data['train']['Sentences']]

    print("Done")
    pdf = pd.DataFrame({
            "label": labels,
            "sent_embd": sent_embd
        })
    return pdf
