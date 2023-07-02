import pandas as pd
import numpy as np
from tqdm.auto import tqdm

from inltk.inltk import setup,tokenize,get_embedding_vectors, get_sentence_encoding

if False:
    setup('ne')


def getWordEmbeddings(data = None, max_len = 110):
    if not data:
        raise("Missing data")

    # print("Getting Sentence Embeddings")
    word_embd = get_embedding_vectors(data,'ne')

    # print("Adding padding vectors to embedding vectors")
    for i in range(len(word_embd)):
        tmp = np.zeros((max_len,400)) 
        tmp -= 0.000001
        tmp[:word_embd[i].shape[0],:] = word_embd[i]
        word_embd[i] = tmp
        # print(tmp.shape)

    # print("Done")
    return np.array(word_embd)

def getSentenceEmbeddings(data = None, max_len = 110):  
    if not data:
        raise("Missing data")

    # tokens = tokenize(data,'ne')
    #
    # print("padding to generate vectors")
    # for i in range(len(data)):
    #     data[i] += " " + " ".join(["-" for i in range(max_len - len(tokens[i]))])
    #   
    # print(type(data))
    # print("Getting Sentence Embeddings")
    # sent_embd = get_sentence_encoding(data,'ne')

    # print("Done")
    return [get_sentence_encoding(text,'ne') for text in data]


if __name__ == '__main__':
    from datasets import load_dataset
    data = load_dataset('raygx/NepCov19TweetsPlus')
    data = data.rename_column(original_column_name='Sentences',new_column_name='text')
    data = data.rename_column(original_column_name='Sentiment',new_column_name='label')
    print("Word Embeddings:\n",getWordEmbeddings(data['train'][:100]['text'])[:2])
    print("Sentence Embeddings:\n",getSentenceEmbeddings(data['train'][:100]['text']))
