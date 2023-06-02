import os
from flask import Flask, request
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

app = Flask(__name__)

MODEL_PATH = "./my_model2"
tokenizer = None
model = None

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

if os.path.exists(MODEL_PATH):
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH)
    model.to(device)

@app.route('/generate', methods=['POST'])
def generate_text():
    data = request.get_json()

    if 'input_text' not in data:
        return "Missing parameter: input_text", 400

    input_text = "USER: " + data['input_text']

    inputs = tokenizer.encode(input_text, return_tensors='pt').to(device)

    try:
        with torch.no_grad():
            outputs = model.generate(inputs, max_length=100)

        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_text = generated_text[len(input_text):]
        generated_text = generated_text.replace("ASSISTANT: ", "")
        generated_text = generated_text.replace("\n", " ").replace("\r", " ")
        generated_text = ' '.join(generated_text.split())
    except Exception as e:
        return f"Error in generating text: {str(e)}", 500

    return generated_text

if __name__ == '__main__':

    if not os.path.exists(MODEL_PATH):
        tokenizer = AutoTokenizer.from_pretrained("openaccess-ai-collective/manticore-13b")
        model = AutoModelForCausalLM.from_pretrained("openaccess-ai-collective/manticore-13b")
        model.to(device)
        tokenizer.save_pretrained(MODEL_PATH)
        model.save_pretrained(MODEL_PATH)

    app.run(host='0.0.0.0', port=5000)
