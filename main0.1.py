# main.py
from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM
import os
from utils import load_tokenizer, encode_input

app = Flask(__name__)

MODEL_PATH = "./my_model2"
tokenizer = None
model = None

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()

    if 'input_text' not in data:
        return jsonify({"error": "Missing parameter: input_text"}), 400

    input_text = data['input_text']
    inputs = encode_input(tokenizer, input_text, device)

    with torch.no_grad():
        outputs = model.generate(inputs, max_length=100)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'generated_text': generated_text})

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        tokenizer = load_tokenizer(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
        model.to(device)

    app.run(host='0.0.0.0', port=5000)
