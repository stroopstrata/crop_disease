from flask import Flask, request, jsonify, render_template
from PIL import Image
import numpy as np
import tensorflow as tf

app = Flask(__name__)

# Load your trained crop disease model
model = tf.keras.models.load_model(r'model\potato_disease_model.keras')

# Preprocess the image function (updated)
def preprocess_image(image):
    image = image.convert('RGB')
    image = image.resize((150, 150))  # Resize as per training size
    image = np.array(image) / 255.0   # Rescale by 1./255
    image = np.expand_dims(image, axis=0)  # Add batch dimSension
    return image

@app.route('/')
# def main_page():
#     return render_template('main.html')
# @app.route('/index')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['image']
    
    # Convert the file to an image
    image = Image.open(file)
    
    # Preprocess the image
    processed_image = preprocess_image(image)
    
    # Run the model prediction
    prediction = model.predict(processed_image)
    predicted_class = int(np.round(prediction[0][0]))  # Since it's binary, round to 0 or 1
    
    # Map the predicted class (0 or 1) to 'Healthy' or 'Diseased'
    class_labels = ['Diseased', 'Healthy']  # Assuming 0: Healthy, 1: Diseased
    predicted_label = class_labels[predicted_class]
    
    return jsonify({'prediction': predicted_label})

if __name__ == '__main__':
    app.run(debug=True)
