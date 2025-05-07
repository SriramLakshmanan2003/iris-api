from flask import Flask, request, jsonify
import joblib
import numpy as np

# Load the trained model
model = joblib.load("iris_model.pkl")

# Create Flask app
app = Flask(__name__)

@app.route('/')
def home():
    return "Welcome to Iris Flower Prediction API!"

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Try to parse JSON data first; fall back to form data
        if request.is_json:
            data = request.get_json()
        else:
            data = request.form

        # Extract and validate features
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

    # Make prediction
    prediction = model.predict([features])[0]

    # Convert prediction to class name if it's a numeric label
    classes = ['Setosa', 'Versicolor', 'Virginica']
    try:
        result = classes[int(prediction)]
    except (ValueError, IndexError):
        result = prediction  # If already a class name, just return it

    return jsonify({"prediction": result})

if __name__ == '__main__':
    app.run(debug=True)
