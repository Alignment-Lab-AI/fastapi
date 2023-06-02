from flask import Flask, request, jsonify
import torch
from transformers import AutoModelForCausalLM
import os
from tokenizer_utils import get_tokenizer, tokenize_text

app = Flask(__name__)

MODEL_PATH = "./my_model2"
tokenizer = None
model = None

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if os.path.exists(MODEL_PATH):
    tokenizer = get_tokenizer(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.to(device)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()

    if 'input_text' not in data:
        return jsonify({"error": "Missing parameter: input_text"}), 400

    input_text = data['input_text']

    inputs = tokenizer.encode(input_text, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model.generate(inputs, max_length=100)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'generated_text': generated_text})

@app.route('/tokenize', methods=['POST'])
def tokenize_endpoint():
    data = request.get_json()

    if 'input_text' not in data:
        return jsonify({"error": "Missing parameter: input_text"}), 400

    if 'output_path' not in data:
        return jsonify({"error": "Missing parameter: output_path"}), 400

    input_text = data['input_text']
    output_path = data['output_path']

    tokenize_text(tokenizer, input_text, output_path)  # use the function here

    return jsonify({'message': f'Tokenized data saved to {output_path}'})

if __name__ == '__main__':

    if not os.path.exists(MODEL_PATH):
        tokenizer = get_tokenizer(MODEL_PATH)
        model = AutoModelForCausalLM.from_pretrained("openaccess-ai-collective/manticore-13b")
        model.to(device)

    app.run(host='0.0.0.0', port=5000)
