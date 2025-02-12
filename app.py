import streamlit as st
from PIL import Image
import torch
import io
import tifffile
import numpy as np
import cv2
import json
import re
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from landingai.predict import Predictor
from landingai.visualize import overlay_predictions
from io import BytesIO
from sklearn.linear_model import LinearRegression
import os

# Load models
MODEL_PATH = "model4.pt"
LINEAR_MODEL_PATH = "finalized_LINEAR_model.sav"

# Ensure models are loaded only once
@st.cache_resource
def load_models():
    model = torch.load(MODEL_PATH, map_location=torch.device('cpu'),weights_only=False)
    model.eval()
    linear_model = pickle.load(open(LINEAR_MODEL_PATH, 'rb'))
    return model, linear_model

model, linear_model = load_models()

# Image processing functions
def process_image(image):
    """Normalize image to [0, 1] range."""
    min_val = np.min(image)
    max_val = np.max(image)
    normalized_image = (image - min_val) / (max_val - min_val)
    return normalized_image

def generate_image(model, input_image):
    with torch.no_grad():
        generated_image = model(input_image)
        generated_image = (generated_image >= 0.5).float()
    return generated_image.squeeze().cpu().numpy()

def flare_extractor(img):
    x, y, c = img.shape
    real_img = np.zeros((x, y, c))
    red = img[:, :, 0]
    green = img[:, :, 1]
    blue = img[:, :, 2]
    only_blue_carbon = np.uint8(blue > 150) * np.uint8(green < 120) * np.uint8(red < 120)
    only_blue_carbon_sum = np.sum(only_blue_carbon)
    only_green_carbon = np.uint8(blue < 120) * np.uint8(green > 150) * np.uint8(red < 120)
    only_green_carbon_sum = np.sum(only_green_carbon)
    only_red_carbon = np.uint8(blue < 120) * np.uint8(green < 120) * np.uint8(red > 180)
    only_red_carbon_sum = np.sum(only_red_carbon)
    only_yellow_carbon = np.uint8(blue < 120) * np.uint8(green > 180) * np.uint8(red > 180)
    only_yellow_carbon_sum = np.sum(only_yellow_carbon)
    real_img[:, :, 0] = np.logical_xor(only_red_carbon, only_yellow_carbon) * 255
    real_img[:, :, 1] = np.logical_xor(only_green_carbon, only_yellow_carbon) * 255
    real_img[:, :, 2] = only_blue_carbon * 255
    out = np.uint8(real_img)
    return out, only_red_carbon_sum, only_green_carbon_sum, only_blue_carbon_sum, only_yellow_carbon_sum

# Streamlit app
def main():
    st.title("Nature Eye Labs: Crisis Management and Planning Platform")
    st.image("WhatsApp Image 2024-09-02 at 15.17.35_cffe6bdc.jpg", width=600)  # Replace with your logo path

    # Rapid Flood Mapping
    st.header("Rapid Flood Mapping")
    uploaded_file = st.file_uploader("Choose a TIFF image...", key="1", type="tif")
    if uploaded_file is not None:
        input_image = tifffile.imread(uploaded_file)
        input_image = process_image(input_image)
        input_image = torch.from_numpy(input_image).permute(2, 0, 1).unsqueeze(0).float()
        output_image = generate_image(model, input_image)
        output_image = np.array(output_image)
        prec = np.sum(output_image) / (128 * 128)
        st.write(f"Area of water: {round(prec * 90)} m^2")
        st.write(f"Area of land: {round((1 - prec) * 90)} m^2")
        output_pil_image = Image.fromarray((output_image * 255).astype('uint8'))
        st.image(output_pil_image, caption='Generated Image', use_column_width=True)

    # Mangrove Detection
    st.header("Mangrove Detection")
    uploaded_file = st.file_uploader("Choose an image...", key="2", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        endpoint_id = os.getenv("LANDINGAI_ENDPOINT_ID")  # Use environment variable
        api_key = os.getenv("LANDINGAI_API_KEY")  # Use environment variable
        image = Image.open(uploaded_file)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        predictor = Predictor(endpoint_id, api_key=api_key)
        predictions = predictor.predict(image)
        color_map = {
            "mangrove": "pink",
            "land": "yellow",
            "mangrove tree": "purple",
            "building": "green",
            "water": "blue",
        }
        options = {"color_map": color_map}
        overlay = overlay_predictions(predictions, image, options=options)
        st.image(overlay, caption='Generated Image', use_column_width=True)

    # Methane Emission Detection
    st.header("Methane Emission Detection")
    uploaded_file = st.file_uploader("Choose an image...", key="3", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image, caption='Uploaded Image', use_column_width=True)
        flare, red, green, blue, yellow = flare_extractor(image)
        flare = Image.fromarray(flare)
        st.image(flare, caption='Generated Image', use_column_width=True)
        st.write(f"Bloom intensity: {linear_model.predict([[red, green, blue, yellow]])}")

if __name__ == "__main__":
    main()