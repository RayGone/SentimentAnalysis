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
  
  import tensorflow as tf



class GPTEmbedding(tf.keras.layers.Layer):
  def __init__(self, model,tokenizer, max_token_len=128,padding='max_length',truncation=True):
    super().__init__()
    self.embedding = model
    self.tokenizer = tokenizer
    self.tokenizer.padding_side = 'left'  ## This is must in case of GPT2
    self.max_token = max_token_len
    self.padding = padding
    self.truncation = truncation
    self.trainable=False

  def call(self, x):
    ## Standard Final Context Approach
    embedding = self.embedding(
                      self.tokenizer(x,padding=self.padding,truncation=self.truncation,max_length=self.max_token,return_tensors='tf')
                      )[0][:,-1:,:]
    return embedding
  
class GPTEmbeddingTrailingContext(GPTEmbedding):
  def call(self, x):
    trailing_context_size = 4
    ## Trailing Context Approach
    embedding = self.embedding(
                      self.tokenizer(x,padding=self.padding,truncation=self.truncation,max_length=self.max_token,return_tensors='tf')
                      )[0][:,-trailing_context_size:,:]
    
    return tf.reduce_logsumexp(embedding,axis=1) * 0.1