import tensorflow as tf
import transformers
from transformers import AutoModel, pipeline
from tokenizers import Tokenizer
from datasets import load_dataset

print(
    tf.config.list_physical_devices()
)
pipe = pipeline('text-generation',
                model = "raygx/GPT2-Nepali-Casual-LM", 
                tokenizer="raygx/GPT2-Nepali-Casual-LM",max_length=100)

print(pipe("स्वास्थ्य मन्त्रालयले पीसीआर"))