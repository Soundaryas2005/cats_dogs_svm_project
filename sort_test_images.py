import os
import cv2
import numpy as np
from tqdm import tqdm
import tflearn
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.estimator import regression
import shutil

# Constants
IMG_SIZE = 50
MODEL_NAME = 'dogvscats-cnn-0.001-model.model'
TEST_DIR = 'images/test'
CAT_DIR = 'images/train/cat'
DOG_DIR = 'images/train/dog'

# Create folders if they don't exist
os.makedirs(CAT_DIR, exist_ok=True)
os.makedirs(DOG_DIR, exist_ok=True)

# CNN Model definition (same as in cnn_classifier.py)
convnet = input_data(shape=[None, IMG_SIZE, IMG_SIZE, 1], name='input')
convnet = conv_2d(convnet, 32, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = conv_2d(convnet, 64, 5, activation='relu')
convnet = max_pool_2d(convnet, 5)
convnet = fully_connected(convnet, 1024, activation='relu')
convnet = dropout(convnet, 0.8)
convnet = fully_connected(convnet, 2, activation='softmax')
convnet = regression(convnet, optimizer='adam',
                     learning_rate=0.001, loss='categorical_crossentropy', name='targets')

model = tflearn.DNN(convnet)
model.load(MODEL_NAME)

# Predict and sort images
for img_file in tqdm(os.listdir(TEST_DIR)):
    path = os.path.join(TEST_DIR, img_file)
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Skipped {img_file}")
        continue
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img.reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
    prediction = model.predict(img)[0]

    if prediction[0] > prediction[1]:  # cat
        shutil.copy(path, os.path.join(CAT_DIR, img_file))
    else:
        shutil.copy(path, os.path.join(DOG_DIR, img_file))
