import tensorflow as tf

'''
    model: huggingface transformer model (without any heads)
    tokenizer: huggingface tokenizer respcetive to the model
    max_token_len: maximum token that tokenizer should output
    padding: padding strategy to be used by tokenizer
    truncation: truncation strategy to be used by tokenizer
'''
class BERTEmbedding(tf.keras.layers.Layer):
  def __init__(self, model,tokenizer = None, max_token_len=128,padding='max_length',truncation=True):
    super().__init__()
    self.embedding = model
    self.tokenizer = tokenizer
    self.max_token = max_token_len
    self.padding = padding
    self.truncation = truncation
    self.trainable=False

  def call(self, x):
    return self.embedding(self.tokenizer(x,padding=self.padding,truncation=self.truncation,max_length=self.max_token,return_tensors='tf'))[1]