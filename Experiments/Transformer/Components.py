import numpy as np
import tensorflow as tf

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

  positions = np.arange(length)[:, np.newaxis]              # (seq, 1)
  depths = np.arange(depth)[np.newaxis, :]/depth            # (1, depth)

  angle_rates = 1 / (10000**depths)                         # (1, depth)
  angle_rads = positions * angle_rates + angle_rates*0.1    # (pos, depth)

  sin = np.sin(angle_rads)
  cos = np.cos(angle_rads)
  pos_encoding = np.concatenate([sin, cos], axis=-1)

  ipos_encoding = np.zeros(pos_encoding.shape)
  ipos_encoding[:, ::2] = sin
  ipos_encoding[:, 1::2] = cos
  if merge=='concat':
        return tf.cast(pos_encoding, dtype=tf.float32)
        print("Concatanation",str(pos_encoding[:2]),pos_encoding.shape)
  elif merge=='add':
        return tf.cast((pos_encoding+ipos_encoding)/2, dtype=tf.float32)
  else:
        return tf.cast(ipos_encoding, dtype=tf.float32)
        print("Interleaving",str(ipos_encoding[:2]),ipos_encoding.shape)

class PositionalEmbedding(tf.keras.layers.Layer):
  def __init__(self, vocab_size, d_model, context_length=2048,pos_enc_merge='add'):
    super().__init__()
    self._name='PosEmbd'
    self.d_model = d_model
    self.embedding = tf.keras.layers.Embedding(vocab_size, d_model, mask_zero=True)
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

  def build(self,input_shape):
    self.mha._build_from_signature(tf.TensorShape(input_shape),tf.TensorShape(input_shape),tf.TensorShape(input_shape))

class GlobalSelfAttention(BaseAttention):
  def call(self, x):
    attn_output = self.mha(
        query=x,
        value=x,
        key=x)
    
    self.attention_output = attn_output
    x = self.add([x, attn_output])
    x = self.layernorm(x)
    return x

class LocalizedSelfAttention(BaseAttention):
  def __init__(self, num_window=8,**kwargs):
    super().__init__(**kwargs)
    self._name='Local_Self_Attention'
    self.num_window = num_window
    self.reshape_orig_layer = None
    self.reshape_window_layer = None
    self.concat_layer = tf.keras.layers.Concatenate(axis=1)

  def call(self, x):                                 
    self.attention_output = self.concat_layer([
      self.mha(key=t,query=t,value=t)
      for t in tf.split(x,num_or_size_splits=self.num_window,axis=1)
    ])
    x = self.add([x, self.attention_output])
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

class Encoder(tf.keras.layers.Layer):
  def __init__(self, *, num_layers, d_model, GSA_num_heads,LSA_num_window,LSA_num_heads,dff, dropout_rate=0.1):
    super().__init__()
    
    self.d_model = d_model
    self.num_layers = num_layers if num_layers else 1

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
    for i in range(self.num_layers):
        x = self.layer_norm(self.add([self.lsa_enc_layers[i](x),self.gsa_enc_layers[i](x)]))    
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
      
      
      