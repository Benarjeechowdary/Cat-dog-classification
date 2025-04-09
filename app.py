from flask import Flask, render_template, request
import cv2
import os
import numpy as np
import joblib
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
IMAGE_SIZE = 64

# Load SVM model
model = joblib.load('svm_cat_dog_model.pkl')

def predict_image(img_path):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    img = img.flatten().reshape(1, -1)
    pred = model.predict(img)
    return 'Cat 🐱' if pred[0] == 0 else 'Dog 🐶'

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    image_path = None

    if request.method == 'POST':
        file = request.files['image']
        if file:
            filename = secure_filename(file.filename)
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(image_path)

            prediction = predict_image(image_path)

    return render_template('index.html', prediction=prediction, image_path=image_path)

if __name__ == '__main__':
    app.run(debug=True)
