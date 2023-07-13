import datasets
import numpy as np
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
import tensorflow as tf

# import torch
# print(torch.cuda.is_available())
# import tensorflow as tf
# print(tf.config.list_physical_devices('GPU'))
# exit()

np.random.seed(5)

data = datasets.Dataset.load_from_disk("NLTK\sentence_embeddings")
print(data)

train = np.array(data['sent_embd'])
algo = KMeans(n_clusters=3)
kmeans = algo.fit(train)

print(set(kmeans.labels_))
# train = data.filter(lambda x: x['label']==0)['sent_embd']
# print(train)
# predict = kmeans.score(train)

# print(predict)