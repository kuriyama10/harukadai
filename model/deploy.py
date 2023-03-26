import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request

# Load model
model = tf.keras.models.load_model("model.h5")

# Create Flask app
app = Flask(__name__)

# Define prediction endpoint
@app.route("/predict", methods=["POST"])
def predict():
    # Get image data from request
    image = request.json["image"]

    # Preprocess image data
    image = np.array(image).reshape((1, 28, 28, 1)) / 255.0

    # Make prediction
    prediction = model.predict(image).tolist()

    # Return prediction as JSON response
    response = {"prediction": prediction}
    return jsonify(response)

# Run Flask app
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=80)
