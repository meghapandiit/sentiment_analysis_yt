import requests
import pickle
from flask import Flask, jsonify, request

# Create a Flask app
app = Flask(__name__)

# Direct download URL for your model
model_url = 'https://drive.google.com/uc?export=download&id=1-EDNlqPttbXC1x7bxz72Myac-YSflc8S'
model_path = 'model.pkl'

# Download the model from Google Drive
def download_model():
    response = requests.get(model_url)
    if response.status_code == 200:
        # Check if the response is a valid file by looking at the content type
        if 'text/html' in response.headers.get('Content-Type'):
            print("Error: Received an HTML page, likely a Google Drive warning or error.")
            print(response.text)  # Print the content to debug
            raise Exception("Model download failed due to an error page.")
        with open(model_path, 'wb') as file:
            file.write(response.content)
        print("Model downloaded successfully.")
    else:
        print(f"Failed to download model, status code: {response.status_code}")
        raise Exception("Model download failed")

# Load the model
def load_model():
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully.")
    return model

# Download and load the model at startup
download_model()
model = load_model()

@app.route('/')
def home():
    return "Model is ready to use."

# Example route to use the model (modify as needed)
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    # Replace with your prediction logic
    result = model.predict([data['features']])
    return jsonify({'prediction': result.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
