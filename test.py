import tensorflow as tf
import transformers
from transformers import AutoModel, pipeline
from tokenizers import Tokenizer
from datasets import load_dataset

# print(
#     tf.config.list_physical_devices()
# )
# pipe = pipeline('text-generation',
#                 model = "raygx/GPT2-Nepali-Casual-LM", 
#                 tokenizer="raygx/GPT2-Nepali-Casual-LM",max_length=100)

# print(pipe("स्वास्थ्य मन्त्रालयले पीसीआर"))

# print("Loading LSTM model")
# model = tf.keras.models.load_model("saved_models/LSTM_4_SA_with_aug")
# print(model.get_layer('embedding').get_weights()[0].shape)

data = load_dataset("cc100", lang="ne")
print(data)