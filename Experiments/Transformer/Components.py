import numpy as np
import tensorflow as tf
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

##------------
##------Position----------------
##----------------------

'''
    merge: 'interleave' or 'concat' or 'add' or None : defaults: 'interleave'
        -> when 'concat' combines [sin,sin,...,cos,cos,...]
        -> when 'interleave' combines [sin,cos,sin,cos......]
        -> when 'add' it adds 'interleave' and 'concat'
'''
def PositionalEncoding(seq_length=2048,feature_depth=512,merge='interleave'):
      depth = feature_depth/2
      length = seq_length

      positions = np.arange(length)[:, np.newaxis]      # (seq, 1)
      depths = np.arange(depth)[np.newaxis, :]/depth    # (1, depth)

      angle_rates = 1 / (10000**depths)                 # (1, depth)
      angle_rads = positions * angle_rates + 0.0001     # (pos, depth)

      sin = np.sin(angle_rads)
      cos = np.cos(angle_rads)
      pos_encoding = np.concatenate([sin, cos], axis=-1)

      ipos_encoding = np.zeros(pos_encoding.shape)
      ipos_encoding[:, ::2] = sin
      ipos_encoding[:, 1::2] = cos
      if merge=='concat':
            return tf.cast(pos_encoding, dtype=tf.float32)
            print("Concatanation",str(pos_encoding[:2]),pos_encoding.shape)
      else:
            return tf.cast(ipos_encoding, dtype=tf.float32)
            print("Interleaving",str(ipos_encoding[:2]),ipos_encoding.shape)

class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model, context_length=2048,pos_enc_merge='interleave'):
    super().__init__()
    self._name='PosEmbd'
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model,
                                               mask_zero=True)
    self.pos_encoding = PositionalEncoding(seq_length=context_length, feature_depth=d_model,merge=pos_enc_merge)

  def compute_mask(self, *args, **kwargs):
    return self.embedding.compute_mask(*args, **kwargs)

  def call(self, x):
    length = tf.shape(x)[1]
    x = self.embedding(x)
    # This factor sets the relative scale of the embedding and positonal_encoding.
    x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
    x = x + self.pos_encoding[tf.newaxis, :length, :]
    return x

##------------
##------Attention----------------
##----------------------
class BaseAttention(tf.keras.layers.Layer):
  def __init__(self, **kwargs):
    super().__init__()
    self.mha = tf.keras.layers.MultiHeadAttention(**kwargs)
    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()


"""LocalizedSelfAttention
    Note: When specifiying num_heads for LocalizedSelfAttention, specify half of what you want.
    if you want 8 heads, set num_heads to 4. Because there is two instances of self attention.
"""

class LocalGlobalSelfAttention(tf.keras.layers.Layer):
  def __init__(self,num_heads, key_dim, dropout, num_window=8):
    super().__init__()
    self._name='LocalGlobal_Self_Attention'
    
    self.local_mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,key_dim=key_dim,dropout=dropout)
    self.global_mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads,key_dim=key_dim,dropout=dropout)

    self.layernorm = tf.keras.layers.LayerNormalization()
    self.add = tf.keras.layers.Add()
    
    self.num_window = num_window
    self.concat_layer = tf.keras.layers.Concatenate(axis=1)
    
    self.global_attention_score = None
    self.local_attention_score = None

    
  def call(self, x):
    ##-------------------------------------
    global_attn_output,self.global_attention_score = self.global_mha(
        query=x,
        value=x,
        key=x, return_attention_scores=True)
    
    ##---------------------------------------
    local_attn_output = []
    self.local_attention_score = []
    for t in tf.split(x,num_or_size_splits=self.num_window,axis=1):
      aout, ascore = self.local_mha(key=t,query=t,value=t,return_attention_scores=True)
      local_attn_output.append(aout)
      self.local_attention_score.append(ascore)
      
    local_attn_output = self.concat_layer(local_attn_output)
    
    ##---------------------------------------
    x = self.add([x, global_attn_output,local_attn_output])
    x = self.layernorm(x)
    return x

#-------
##------FeedForward-------------
##----------------------

class FeedForward(tf.keras.layers.Layer):
  def __init__(self, d_model, dff, dropout_rate=0.1):
    super().__init__()
    self.seq = tf.keras.Sequential([
      tf.keras.layers.Dense(dff, activation='relu'),
      tf.keras.layers.Dense(d_model),
      tf.keras.layers.Dropout(dropout_rate)
    ])
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()

  def call(self, x):
    x = self.add([x, self.seq(x)])
    x = self.layer_norm(x) 
    return x

##-------
##------Encoder----------------
##----------------------

class LSA_EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model,num_window, dff, num_heads=2,dropout_rate=0.1):
    super().__init__()

    self._name='Local_Self_Attention_Encoder'
    self.self_attention = LocalizedSelfAttention(
        num_window=num_window,
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x

class GSA_EncoderLayer(tf.keras.layers.Layer):
  def __init__(self,*, d_model, num_heads, dff, dropout_rate=0.1):
    super().__init__()

    self._name='Global_Self_Attention_Encoder'
    self.self_attention = GlobalSelfAttention(
        num_heads=num_heads,
        key_dim=d_model,
        dropout=dropout_rate)

    self.ffn = FeedForward(d_model, dff)

  def call(self, x):
    x = self.self_attention(x)
    x = self.ffn(x)
    return x

"""_summary_

  attn_stack_type: defines how to arrange LSA and GSA; defaults to 'add' [(LSA+GSA),...]; 
                    another option is 'stack': one after another [GSA,LSA,.....,GSA+LSA]
"""
class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, GSA_num_heads,LSA_num_window,LSA_num_heads,dff,attn_stack_type='add', dropout_rate=0.1):
    super().__init__()
    
    self.d_model = d_model
    self.num_layers = num_layers if num_layers else 1
    self.attn_stack_type = attn_stack_type ##By Default Add
    if self.attn_stack_type not in ['add','stack']:
      self.attn_stack_type = 'add'

    self.lsa_enc_layers = [
                            LSA_EncoderLayer(d_model=d_model,
                                num_window=LSA_num_window,
                                num_heads=LSA_num_heads,
                                dff=dff,
                                dropout_rate=dropout_rate) 
                            for _ in range(num_layers)
                        ]
    self.gsa_enc_layers = [
                            GSA_EncoderLayer(d_model=d_model,
                                num_heads=GSA_num_heads,
                                dff=dff,
                                dropout_rate=dropout_rate)                            
                            for _ in range(num_layers)
                        ]
          
    self.add = tf.keras.layers.Add()
    self.layer_norm = tf.keras.layers.LayerNormalization()
    self.dropout = tf.keras.layers.Dropout(dropout_rate)

  def call(self, x): 
    if self.attn_stack_type == 'add' or self.num_layers==1:      
      for i in range(self.num_layers):
          x = self.layer_norm(self.add([self.lsa_enc_layers[i](x),self.gsa_enc_layers[i](x)]))  
    if self.attn_stack_type == 'stack':
      for i in range(self.num_layers - 1):
        x = self.lsa_enc_layers[i](
              self.gsa_enc_layers[i](x)
            )
        
        x = self.layer_norm(self.add([self.lsa_enc_layers[-1](x),self.gsa_enc_layers[-1](x)]))  
        
    return self.dropout(x)


##---------------------------
if __name__ == '__main__':
  import matplotlib.pyplot as plt
  if False: # Positional Encoding and Embedding

    pos_encoding = PositionalEncoding(merge='concat').numpy()
    apos_encoding = PositionalEncoding(merge='add').numpy()
    ipos_encoding = PositionalEncoding().numpy()
    # Plot the dimensions.
    plt.pcolormesh(pos_encoding.T, cmap='RdBu')
    plt.ylabel('Depth')
    plt.xlabel('Position')
    plt.colorbar()
    plt.show()

    plt.pcolormesh(ipos_encoding.T, cmap='RdBu')
    plt.ylabel('Depth')
    plt.xlabel('Position')
    plt.colorbar()
    plt.show()

    plt.pcolormesh(apos_encoding.T, cmap='RdBu')
    plt.ylabel('Depth')
    plt.xlabel('Position')
    plt.colorbar()
    plt.show()
    
    
  import numpy as np
  
  batch = np.ones((2,128,768))
  # print(batch,batch.shape)
  
  if False:    ### testing tf.image.extract
    windows = tf.reshape(batch,(batch.shape[0],batch.shape[1],12,int(768/12)))
    print(windows,windows.shape)
    print(tf.reshape(windows,batch.shape))
    
  if False:    ### testing Localized_Self_Attention_Layer
    LSA = LocalizedSelfAttention(
            key_dim=768,
            dropout=0.15
        )
    
    print(LSA(batch))
      
      
      