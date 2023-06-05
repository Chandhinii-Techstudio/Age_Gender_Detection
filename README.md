## Image classification using CNN Model
This repository contains code for preparing the CATS_DOGS dataset and training a Convolutional Neural Network (CNN) model using Keras.

# Dataset Preparation
Download the CATS_DOGS dataset.
Extract the dataset to a local directory.
Perform image manipulation techniques such as rotation, width shift, height shift, rescale, horizontal flip to preprocess the images.
Split the dataset into training and testing sets.

# CNN Model Architecture
Define a CNN model architecture using Keras.
Specify the number of convolutional layers, pooling layers, and fully connected layers.
Choose appropriate activation functions, filter sizes, and strides for the convolutional layers.
Add dropout layers to prevent overfitting.
Compile the model with a suitable loss function (binary_crossentropy) and optimizer (adam).

# Model Training
Load the preprocessed dataset into the model.
Train the model using the training set.
Monitor the training progress by tracking the loss and accuracy metrics.
Validate the model's performance using the validation set.

# Model Evaluation
Plot graphs to visualize the training and validation metrics over epochs.
Calculate the test accuracy of the trained model using the testing set.
Compare the test accuracy with the desired accuracy.

# Predicting Unseen Images
Use the trained model to predict the classes of new unseen images.
Load the unseen image into the model.
Obtain the corresponding probabilities.

# Results
The trained model achieved a test accuracy of 88% on the CATS_DOGS dataset.
The training and validation graphs demonstrate the model's learning progress.
The model accurately predicts the dog class of new unseen images.
