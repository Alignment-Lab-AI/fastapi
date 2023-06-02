import numpy as np
from transformers import AutoTokenizer

def get_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path)

def tokenize_text(tokenizer, input_text, output_path):
    inputs = tokenizer.encode(input_text, return_tensors='pt')
    np.save(output_path, inputs['input_ids'].numpy())
