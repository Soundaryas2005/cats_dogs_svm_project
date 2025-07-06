% visualize_predictions.m
% This MATLAB script visualizes predictions from SVM model

folder = 'images/test'; % Folder with mixed cat and dog test images
modelFile = 'svm_model.pkl'; % Trained model from Python
imgSize = 128;

% Load Python model
model = py.joblib.load(modelFile);
files = dir(fullfile(folder, '*.jpg'));

figure;
for k = 1:12
    if k > length(files)
        break;
    end
    filename = fullfile(folder, files(k).name);
    img = imread(filename);
    img_resized = imresize(img, [imgSize imgSize]);
    img_norm = double(img_resized) / 255;
    
    % Preprocess for MobileNetV2 compatibility (like in Python)
    img_reshaped = reshape(img_norm, [1 imgSize imgSize 3]);
    feature_extractor = py.importlib.import_module('tensorflow.keras.applications.mobilenet_v2');
    pre_img = feature_extractor.preprocess_input(py.numpy.array(img_reshaped));
    
    % Predict using saved model
    prediction = model.predict(pre_img);
    label = 'Cat';
    if prediction{1} == 1
        label = 'Dog';
    end
    
    subplot(3, 4, k);
    imshow(img);
    title(label);
end
