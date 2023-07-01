import datasets

data = datasets.Dataset.load_from_disk("sentence_embeddings")
print(data[1])