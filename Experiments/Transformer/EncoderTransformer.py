import tensorflow as tf
from Components import PositionalEmbedding, Encoder

"""_summary_

  attn_stack_type: defines how to arrange LSA and GSA; defaults to 'add' [(LSA+GSA),...]; 
                    another option is 'stack': one after another [GSA,LSA,.....,(LSA+GSA)]
"""
class Transformer(tf.keras.Model):
  def __init__(self, *, num_layers, d_model, GSA_num_heads,LSA_num_window,LSA_num_heads,
               dff, vocab_size,num_class=2,attn_stack_type, dropout_rate=0.1):
    super().__init__()

    self.d_model = d_model
    self.num_layers = num_layers if num_layers else 1

    self.pos_embedding = PositionalEmbedding(vocab_size=vocab_size, d_model=d_model)

    self.encoder = Encoder(num_layers=num_layers,d_model=d_model,GSA_num_heads=GSA_num_heads,
                           LSA_num_window=LSA_num_window,LSA_num_heads=LSA_num_heads,
                           dff=dff,attn_stack_type=attn_stack_type,dropout_rate=dropout_rate)
    
    self.dropout = tf.keras.layers.Dropout(dropout_rate)
    self.out = tf.keras.layers.Dense(d_model,activation='gelu',name='feature')
    self.head = tf.keras.layers.Dense(num_class,activation='softmax',name='classification_head')

  def call(self, x):
    # `x` is token-IDs shape: (batch, seq_len)
    x = self.pos_embedding(x)  # Shape `(batch_size, seq_len, d_model)`.

    # Add dropout.
    x = self.dropout(x)
    x = self.encoder(x)
    
    self.last_hidden_state = x
    x = tf.reduce_logsumexp(x,axis=1) * 0.1
    self.pooled_state = self.out(x)
    return self.head(self.pooled_state)  # Shape `(batch_size, seq_len, d_model)`.

if __name__ == '__main__':
    import numpy as np
    
    batch = np.ones((2,128))
    model = Transformer(num_layers=6,d_model=256,GSA_num_heads=8,LSA_num_window=4,LSA_num_heads=2,vocab_size=50000,dff=512)
    print(model(batch))
    print(model.pooled_state)
    print(model.summary())
