import os
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

# Configuration
DATA_DIR = 'Dataset'
CATEGORIES = ['Cat', 'Dog']
IMAGE_SIZE = 64  # Resize images to 64x64

def load_data():
    data = []
    labels = []

    for label, category in enumerate(CATEGORIES):
        path = os.path.join(DATA_DIR, category)
        for img_name in os.listdir(path):
            try:
                img_path = os.path.join(path, img_name)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Convert to grayscale
                img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))   # Resize
                data.append(img.flatten())  # Flatten 2D image to 1D
                labels.append(label)        # 0 for cat, 1 for dog
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")

    return np.array(data), np.array(labels)

def train_svm(X_train, y_train):
    model = SVC(kernel='linear')
    model.fit(X_train, y_train)
    return model

def main():
    print("🔄 Loading and preprocessing data...")
    X, y = load_data()

    print("🔀 Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print("🎯 Training SVM model...")
    model = train_svm(X_train, y_train)

    print("🧪 Evaluating model...")
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"✅ Model Accuracy: {accuracy * 100:.2f}%")

    print("💾 Saving model to 'svm_cat_dog_model.pkl'...")
    joblib.dump(model, 'svm_cat_dog_model.pkl')

    print("🏁 Done!")

if __name__ == '__main__':
    main()
