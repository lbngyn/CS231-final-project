import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import lightgbm as lgb
from skimage.feature import hog
from sklearn.preprocessing import LabelEncoder
import base64
import matplotlib.pyplot as plt


# Load models and other necessary setups
with open('pickle/svm_model_4.pkl', 'rb') as f:
    svm_model = pickle.load(f)
with open('pickle/lightgbm_model_4.pkl', 'rb') as f:
    lightgbm_model = pickle.load(f)
with open('pickle/scaler_4.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('pickle/pca_4.pkl', 'rb') as f:
    pca = pickle.load(f)
with open('pickle/encoder_4.pkl', 'rb') as f:
    encoder = pickle.load(f)
with open('pickle/rf_model_4.pkl', 'rb') as f:
    rf_model = pickle.load(f)
    
def compute_color_histogram(image, bins=(8, 8, 8)):
    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Compute the color histogram
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    # Normalize the histogram
    hist = cv2.normalize(hist, hist).flatten()
    return hist
    
def preprocess_image(image_file):
    image = Image.open(image_file)
    image = np.array(image).astype(np.uint8)
    image = cv2.resize(image, (64, 64))
    grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Extract HOG features
    hog_features = hog(grey_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, block_norm='L2')
    # Extract color histogram features
    color_hist = compute_color_histogram(image)
    # Concatenate HOG and color histogram features
    combined_features = np.hstack((hog_features, color_hist.flatten()))
    return combined_features

def predict(features):
    features = np.array(features).reshape(1, -1)
    features_scaled = scaler.transform(features)
    features_pca = pca.transform(features_scaled)
    svm_prediction = svm_model.predict(features_pca)
    lightgbm_prediction = lightgbm_model.predict(features_pca)
    rf_prediction = rf_model.predict(features_pca)
    svm_label = encoder.inverse_transform(svm_prediction)[0]
    lightgbm_label = encoder.inverse_transform(lightgbm_prediction)[0]
    rf_label = encoder.inverse_transform(rf_prediction)[0]
    return svm_label, lightgbm_label, rf_label

def compute_color_histogram(image, bins=(8, 8, 8)):
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_image], [0, 1, 2], None, bins, [0, 180, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    return hist

def display_hog(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    features, hog_image = hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=True, transform_sqrt=True, block_norm='L2')
    return hog_image

def display_color_hist(hist):
    color = ('b', 'g', 'r')
    for i, col in enumerate(color):
        plt.plot(hist[i], color=col)
    plt.xlim([0, 256])
    plt.title('HSV Color Histogram')
    return plt

# Custom CSS to use the blurred image as background
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/png;base64,{encoded_string}");
            background-size: cover;
            background-position: center center;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Adding the background from local image
add_bg_from_local(r'F:\Ky4\CV\cv\background.png')

# Streamlit application layout
st.title("Flower Image Classification Demo")
st.write("Upload multiple images of flowers, and the model will predict the flower type for each using SVM, LightGBM and RF.")

uploaded_files = st.file_uploader("Choose images...", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if uploaded_files:
    for uploaded_file in uploaded_files:
        # Setup columns for layout
        col1, col2 = st.columns(2)

        # Column for the uploaded image
        with col1:
            st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)

        # Process and predict
        features = preprocess_image(uploaded_file)
        svm_label, lightgbm_label, rf_label = predict(features)

        # Column for predictions
        with col2:
            st.write(f"SVM Prediction: {svm_label}")
            st.write(f"LightGBM Prediction: {lightgbm_label}")
            st.write(f"RF Prediction: {rf_label}")
