Cats vs Dogs Image Classifier (CNN + SVM)

This project implements a **cats vs dogs image classification** system using a hybrid model:
- CNN (EfficientNet)** for feature extraction
- Support Vector Machine (SVM)** for final classification

It was developed as part of my internship with **Prodigy Infotech** to demonstrate efficient and accurate binary image classification using deep learning and machine learning techniques.

---

## 🔍 Project Highlights

- **Input**: Raw images of cats and dogs
- **Model**: EfficientNet pre-trained CNN + SVM classifier
- **Accuracy**: Achieves over **78% validation accuracy**
- **Output**: Predictions and sorted image folders
- **Optional Visualization**: Includes `visualize_predictions.m` for MATLAB

---

# Technologies Used

- **Python 3**
- **TensorFlow** & **TFLearn**
- **OpenCV** for image processing
- **scikit-learn** (SVM)
- **MATLAB** (optional visualization)
- **Git** & **GitHub** for version control

---

# Project Structure
cats_dogs_svm_project/
│
├── cnn_classifier.py # Train CNN (EfficientNet) for feature extraction
├── svm_from_cnn.py # Train and evaluate SVM on CNN features
├── sort_test_images.py # Sort test images based on prediction
├── visualize_predictions.m # MATLAB script for plotting predictions
├── .gitignore # Ignored large files and folders



