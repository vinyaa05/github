import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow warnings

from flask import Flask, request, render_template
from tensorflow.keras.models import load_model  # type: ignore
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load the trained model
model = load_model(r"C:\Users\vinya\OneDrive\Downloads\plant_disease_model (1).h5")

# Manually define the class labels (ensure these match your trained model's labels)
class_labels = ['Healthy', 'Diseased']

# Initialize the LabelEncoder and set the classes
le = LabelEncoder()
le.fit(class_labels)  # Fit the label encoder with the manually defined labels

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return "No file part"

    file = request.files['image']  # Change 'file' to 'image'
    
    if file.filename == '':
        return "No selected file"
    
    img_path = os.path.join('uploads', file.filename)
    file.save(img_path)

    # Preprocess the image
    img = cv2.imread(img_path)
    img = cv2.resize(img, (128, 128))
    img = np.expand_dims(img, axis=0)  # Add batch dimension

    # Predict using the model
    prediction = model.predict(img)
    predicted_class = le.inverse_transform([np.argmax(prediction)])[0]  # Get the class name

    return render_template('result.html', prediction=predicted_class)  # Update to render result.html

if __name__ == '__main__':
    app.run(debug=True)

