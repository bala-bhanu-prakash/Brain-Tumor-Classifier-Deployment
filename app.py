import streamlit as st
from azure.cognitiveservices.vision.customvision.prediction import CustomVisionPredictionClient
from msrest.authentication import ApiKeyCredentials
from PIL import Image
import config  # Import your config file

# Initialize Azure Custom Vision client using the variables from config.py
try:
    credentials = ApiKeyCredentials(in_headers={"Prediction-key": config.PREDICTION_KEY})
    predictor = CustomVisionPredictionClient(config.ENDPOINT, credentials)
except Exception as e:
    st.error(f"Error initializing Azure Custom Vision client: {e}")
    st.stop()

# Streamlit app title
st.title("Brain Tumor Prediction App")

# Upload image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image.", use_column_width=True)

        # Get the image contents as a byte array
        image_contents = uploaded_file.getvalue()

        # Classify the image using Azure Custom Vision
        results = predictor.classify_image(config.PROJECT_ID, config.PUBLISH_ITERATION_NAME, image_contents)

        # Display prediction results
        st.write("Results:")
        for prediction in results.predictions:
            st.write(f"{prediction.tag_name}: {prediction.probability * 100:.2f}%")
    except Exception as e:
        st.error(f"Error processing image: {e}")
