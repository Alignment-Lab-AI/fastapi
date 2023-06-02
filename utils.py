# tokenizer_utils.py
from transformers import AutoTokenizer

def load_tokenizer(model_path):
    return AutoTokenizer.from_pretrained(model_path)

def encode_input(tokenizer, input_text, device):
    return tokenizer.encode(input_text, return_tensors='pt').to(device)
