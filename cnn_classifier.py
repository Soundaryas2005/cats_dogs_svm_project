import os
import numpy as np
import cv2
from tqdm import tqdm
from random import shuffle

import tflearn
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, fully_connected, dropout
from tflearn.layers.estimator import regression

# Constants
IMG_SIZE = 50
LR = 1e-3
MODEL_NAME = f'dogvscats-cnn-{LR}-model'

# Paths
TRAIN_DIR = r'images/train/cat'  # Make sure this is only cat images
TEST_DIR = r'images/test'

def label_img(img_name):
    """Label cat images as [1, 0] and others as [0, 1]."""
    word_label = img_name.split('.')[-3]
    return [1, 0] if word_label == 'cat' else [0, 1]

def create_train_data():
    training_data = []
    print("[INFO] Creating training data...")
    for img in tqdm(os.listdir(TRAIN_DIR)):
        label = label_img(img)
        path = os.path.join(TRAIN_DIR, img)
        img_data = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img_data is None:
            print(f"[WARN] Skipped: {img}")
            continue
        img_data = cv2.resize(img_data, (IMG_SIZE, IMG_SIZE))
        training_data.append([np.array(img_data), np.array(label)])
    shuffle(training_data)
    np.save('train_data.npy', training_data)
    print("[INFO] Training data saved to train_data.npy")
    return training_data

# Load or create training data
if not os.path.exists("train_data.npy"):
    train_data = create_train_data()
else:
    print("[INFO] Loading existing training data...")
    train_data = np.load('train_data.npy', allow_pickle=True)

# Optionally reduce training size for faster testing
# train_data = train_data[:500]

print(f"[INFO] Total training samples: {len(train_data)}")

# Preprocess
X = np.array([i[0] for i in train_data]).reshape(-1, IMG_SIZE, IMG_SIZE, 1) / 255.0
y = np.array([i[1] for i in train_data])

# Define CNN model
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

model = tflearn.DNN(convnet, tensorboard_dir='log')

# Train the model
print("[INFO] Starting training...")
model.fit({'input': X}, {'targets': y}, n_epoch=5,
          validation_set=0.1,
          snapshot_step=500,
          snapshot_epoch=True,
          show_metric=True,
          run_id=MODEL_NAME)

# Save the model
model.save(f"{MODEL_NAME}.model")
print("[INFO] Model trained and saved as", f"{MODEL_NAME}.model")