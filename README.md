# Brain-Tumor-Classifier-Deployment

### Part 1: Building the Model Using CNN

#### Overview
This project focuses on building a Convolutional Neural Network (CNN) to classify brain MRI images into two categories: 'Tumor' and 'No Tumor.' The model is trained on a dataset of labeled images and is then deployed using Microsoft Custom Vision AI for testing through a user interface.

#### Libraries Used
- Keras: For building and training the CNN model.
- PIL (Python Imaging Library): For image processing.
- NumPy: For numerical computations and handling arrays.
- Pandas: For data manipulation and analysis.
- Matplotlib: For plotting the model's loss during training.
- scikit-learn: For data preprocessing and model evaluation.
- Gradio: For creating a web-based interface to test the model.

### Methods and Procedure

#### 1. Data Preparation
The code starts by loading images from two directories: one containing MRI images with tumors and the other without tumors. Each image is resized to 128x128 pixels and converted into a NumPy array. The images are labeled using a one-hot encoding method:
- 0: Tumor
- 1: No Tumor

#### 2. Splitting the Data
The images and their corresponding labels are split into training and testing sets using `train_test_split` from scikit-learn.

#### 3. Building the CNN Model
The model is built using Keras' Sequential API. It consists of the following layers:
- Conv2D Layers: For feature extraction using convolutional filters.
- BatchNormalization: For normalizing the inputs to each layer to stabilize the learning process.
- MaxPooling2D: For down-sampling the input representation to reduce its dimensionality.
- Dropout: For regularization to prevent overfitting.
- Flatten: For converting the 2D matrix data to a vector to be passed into the fully connected layers.
- Dense Layers: For classification purposes, where the final layer uses the softmax activation function to output probabilities for the two classes.

#### 4. Compiling the Model
The model is compiled using the Adamax optimizer and categorical crossentropy loss function.

#### 5. Training the Model
The model is trained using the training data, with early stopping implemented to prevent overfitting.

#### 6. Visualizing the Training Process
The training and validation loss are plotted to visualize the model's performance over epochs.

### Creating the Interface Using Gradio

#### Overview
The Gradio library is used to create an interactive web interface for testing the model. Users can upload an MRI image, and the model will classify it as 'Tumor' or 'No Tumor.'

#### 1. Defining the Prediction Function
A function is defined to preprocess the input image, make predictions using the trained model, and return the classification.

#### 2. Creating the Gradio Interface
The interface is built using Gradio, allowing users to interact with the model. Different themes are provided for customization.

### Part 2: Deployment Using Microsoft Custom Vision

After building the model using CNN, I further trained the same set of images in Microsoft Custom Vision. This platform provides an easy way to build, train, and deploy machine learning models for image classification tasks.

#### Steps Followed

1. **Image Upload and Training:**
   - The same set of MRI images, labeled as 'Tumor' and 'No Tumor,' was uploaded to Microsoft Custom Vision.
   - The platform's interface was used to train the model, leveraging Microsoft's cloud infrastructure to optimize the model for accuracy and performance.

2. **Model Deployment:**
   - Once the model was trained, it was deployed as a web service, generating an endpoint that can be accessed via HTTP requests. This allows the model to be integrated into various applications.

3. **Integration with Streamlit:**
   - Using a Streamlit app, I connected the Custom Vision model's endpoint with the Streamlit interface, enabling users to interact with the model directly through a web browser.
   - The Streamlit app allows users to upload MRI images, which are then sent to the Custom Vision endpoint for classification. The results (either 'Tumor' or 'No Tumor') are displayed in real-time.

#### Accessing the Deployed Model
You can access the deployed model and test it yourself through the following link: [Brain Tumor Classifier on Streamlit](https://brain-tumor-classifier-mcv.streamlit.app/).

This deployment demonstrates how Microsoft Custom Vision and Streamlit can be combined to create a user-friendly and accessible AI application.



