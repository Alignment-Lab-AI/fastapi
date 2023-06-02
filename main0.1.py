from flask import Flask, request, jsonify, abort
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import os
import logging
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = Flask(__name__)

limiter = Limiter(app, key_func=get_remote_address)

# Set up logging
logging.basicConfig(filename='api.log', level=logging.INFO)

MODEL_PATH = "./my_model2"
model_name = os.getenv("MODEL_NAME", "openaccess-ai-collective/manticore-13b")

@app.route('/generate', methods=['POST'])
@limiter.limit("10/minute")  # rate limit
def generate_text():
    data = request.get_json()
    logging.info(f"Received data: {data}")

    if 'input_text' not in data:
        abort(400, description="Missing parameter: input_text")

    input_text = data['input_text']

    # Input validation
    if len(input_text) > 500:
        abort(400, description="Input text too long")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.to(device)
    
    inputs = tokenizer.encode(input_text, return_tensors='pt').to(device)

    with torch.no_grad():
        outputs = model.generate(inputs, max_length=100)

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return jsonify({'generated_text': generated_text})

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "OK"})

if __name__ == '__main__':
    if not os.path.exists(MODEL_PATH):
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(model_name)
        tokenizer.save_pretrained(MODEL_PATH)
        model.save_pretrained(MODEL_PATH)
    app.run(host='0.0.0.0', port=int(os.getenv("PORT", 5000)))
