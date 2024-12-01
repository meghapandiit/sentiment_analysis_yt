from flask import Flask, request, jsonify
import pickle

app = Flask(__name__)

# Load model
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return "Model API is running!"

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.json
    comments = data.get('comments', [])
    if not comments:
        return jsonify({"error": "No comments provided"}), 400
    
    # Preprocess and predict sentiment (update as needed)
    predictions = model.predict(comments)  # Adjust as needed for your model's input

    response = [{"comment": c, "sentiment": p} for c, p in zip(comments, predictions)]
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True)
