#app.py

from flask import Flask, request, jsonify
import torch
import numpy as np  # Import np here
from model import NeuralNet
from utils import bag_of_words, stem  # Ensure utils is imported
import json

app = Flask(__name__)

# Load intents and model
with open('intents.json', 'r') as f:
    intents = json.load(f)

# Load model
model = NeuralNet(input_size=120, hidden_size=8, output_size=31)
model.load_state_dict(torch.load("model.pth", weights_only=True))
model.eval()

# Load all_words and tags (defined in train.py)
with open("train_data.json", "r") as f:  # You will need to save all_words and tags here after training
    train_data = json.load(f)
    all_words = train_data["all_words"]
    tags = train_data["tags"]

# Tag responses from intents
def get_response(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            return np.random.choice(intent["responses"])

@app.route('/chatbot', methods=['POST'])
def chatbot_response():
    data = request.get_json()
    user_message = data["message"]

    X = np.array(bag_of_words(user_message, all_words))
    X = torch.from_numpy(X).float()
    output = model(X)
    _, predicted = torch.max(output, dim=0)

    tag = tags[predicted.item()]
    response = get_response(tag)

    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
