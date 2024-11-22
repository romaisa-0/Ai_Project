from flask import Flask, request, render_template, jsonify
import numpy as np
import cv2
from tensorflow.keras.models import load_model

app = Flask(__name__)

# Load your model
model = load_model('mnist_digit_classifier.h5')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file uploaded", 400
    
    file = request.files['file']
    
    # Process the image file
    img_data = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(img_data, cv2.IMREAD_GRAYSCALE)
    
    # Ensure the image was properly decoded
    if img is None:
        return "Invalid image format", 400
    
    img = cv2.resize(img, (28, 28))  # Resize according to your model's input
    img = img.astype('float32') / 255.0
    img = img.reshape(1, 28, 28, 1)  # Adjust based on your model's input shape
    
    # Make prediction
    prediction = model.predict(img)
    predicted_class = int(np.argmax(prediction))
    probabilities = prediction[0].tolist()  # Convert probabilities to list

    # Return JSON with the predicted class and the probabilities
    return jsonify(predicted_class=predicted_class, probabilities=probabilities)

if __name__ == '__main__':
    app.run(debug=True)
