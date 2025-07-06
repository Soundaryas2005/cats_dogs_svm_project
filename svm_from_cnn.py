# svm_from_cnn.py
import numpy as np
import os
import cv2
from tqdm import tqdm
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression

IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = 'dogvscats-cnn-0.001-model.model'

# Recreate the same CNN architecture
def create_cnn_model():
    convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
    convnet = conv_2d(convnet, 32, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = conv_2d(convnet, 64, 5, activation='relu')
    convnet = max_pool_2d(convnet, 5)
    convnet = fully_connected(convnet, 1024, activation='relu')
    convnet = dropout(convnet, 0.8)
    convnet = fully_connected(convnet, 2, activation='softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=LR,
                         loss='categorical_crossentropy', name='targets')
    model = tflearn.DNN(convnet)
    return model

# Load the trained TFLearn model
print("[INFO] Loading CNN model from .model file...")
cnn_model = create_cnn_model()
cnn_model.load(MODEL_NAME)

# Feature extractor: returns class probability of being a cat
def extract_feature(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)) / 255.0
    img = img.reshape(-1, IMG_SIZE, IMG_SIZE, 1)
    prediction = cnn_model.predict(img)
    return prediction[0][0]  # Probability of 'cat'

# Load dataset
def load_dataset(dataset_path):
    features = []
    labels = []
    for label_folder in ['cat', 'dog']:
        folder_path = os.path.join(dataset_path, label_folder)
        label = 0 if label_folder == 'cat' else 1
        if not os.path.exists(folder_path):
            continue
        for img_name in tqdm(os.listdir(folder_path), desc=f"Processing {label_folder} images"):
            img_path = os.path.join(folder_path, img_name)
            feature = extract_feature(img_path)
            if feature is not None:
                features.append([feature])
                labels.append(label)
    return np.array(features), np.array(labels)

# Load features and labels
print("[INFO] Extracting features from training images...")
X, y = load_dataset('images/train')

if len(X) == 0:
    print("[ERROR] No data loaded. Ensure 'images/train/cat' and 'images/train/dog' exist.")
    exit()

# Train/test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Train SVM
print("[INFO] Training SVM classifier...")
svm = SVC(kernel='linear')
svm.fit(X_train, y_train)

# Evaluate
print("[INFO] Evaluating...")
y_pred = svm.predict(X_val)
print("Accuracy:", accuracy_score(y_val, y_pred))
print(classification_report(y_val, y_pred))
