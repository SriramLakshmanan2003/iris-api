from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib

# Load the trained model
model = joblib.load("iris_model.pkl")

# Create Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

@app.route('/')
def home():
    return "Welcome to Iris Flower Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse input JSON or form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        # Extract and validate input values
        features = [
            float(data['sepal_length']),
            float(data['sepal_width']),
            float(data['petal_length']),
            float(data['petal_width'])
        ]
    except KeyError as e:
        return jsonify({"error": f"Missing field: {e.args[0]}"}), 400
    except ValueError:
        return jsonify({"error": "All inputs must be numeric."}), 400

    # Predict using the loaded model
    prediction = model.predict([features])[0]

    # Map prediction index to class name (if needed)
    classes = ['Setosa', 'Versicolor', 'Virginica']
    try:
        result = classes[int(prediction)]
    except (ValueError, IndexError):
        result = prediction  # Already a class name or unexpected label

    return jsonify({"prediction": result})

# Only run this locally. In production (e.g., Render), Gunicorn will be used.
if __name__ == '__main__':
    app.run()
