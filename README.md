# Brain-Tumor-Classifier-Deployment
Developed a machine learning model using CNN to classify brain MRI images as 'tumor' or 'no tumor'. Created a user interface for testing and deployed the model using Microsoft Custom Vision AI

Part 1: Building the Model Using CNN
Overview
This project focuses on building a Convolutional Neural Network (CNN) to classify brain MRI images into two categories: 'Tumor' and 'No Tumor.' The model is trained on a dataset of labeled images and is then deployed using Microsoft Custom Vision AI for testing through a user interface.

Libraries Used
Keras: For building and training the CNN model.
PIL (Python Imaging Library): For image processing.
NumPy: For numerical computations and handling arrays.
Pandas: For data manipulation and analysis.
Matplotlib: For plotting the model's loss during training.
scikit-learn: For data preprocessing and model evaluation.
Gradio: For creating a web-based interface to test the model.

Methods and Procedure

Data Preparation
The code starts by loading images from two directories: one containing MRI images with tumors and the other without tumors. Each image is resized to 128x128 pixels and converted into a NumPy array. The images are labeled using a one-hot encoding method:

0: Tumor
1: No Tumor

 Splitting the Data
The images and their corresponding labels are split into training and testing sets using train_test_split from scikit-learn.

Building the CNN Model
The model is built using Keras' Sequential API. It consists of the following layers:

Conv2D Layers: For feature extraction using convolutional filters.
BatchNormalization: For normalizing the inputs to each layer to stabilize the learning process.
MaxPooling2D: For down-sampling the input representation to reduce its dimensionality.
Dropout: For regularization to prevent overfitting.
Flatten: For converting the 2D matrix data to a vector to be passed into the fully connected layers.
Dense Layers: For classification purposes, where the final layer uses the softmax activation function to output probabilities for the two classes.

 Compiling the Model
The model is compiled using the Adamax optimizer and categorical crossentropy loss function.

Creating the Interface Using Gradio
The Gradio library is used to create an interactive web interface for testing the model. Users can upload an MRI image, and the model will classify it as 'Tumor' or 'No Tumor'.

Part 2: Deployment using Microsoft Custom Vision

