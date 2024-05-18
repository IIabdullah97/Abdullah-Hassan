# -*- coding: utf-8 -*-
"""
Created on Thu May  16 20:11:05 2024
@author: bih13
"""

import pandas as pd
import streamlit as st
import requests
import joblib

@st.cache(allow_output_mutation=True)
def download_file(url, filename):
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception(f"Error downloading file: {url}")
    with open(filename, "wb") as file:
        file.write(response.content)
    return filename

# URLs of the files in the GitHub repository
pipeline_url = "https://raw.githubusercontent.com/IIabdullah97/Abdullah-Hassan/main/Vectorizer.pkl"
model_url = "https://raw.githubusercontent.com/IIabdullah97/Abdullah-Hassan/main/Model.pkl"
image_url = "https://miro.medium.com/v2/resize:fit:693/0*u_3GNniqZ6e7DSFK.png" 
image_url2 = "https://miro.medium.com/v2/resize:fit:1400/1*_igArwmR7Pj_Mu_KUGD1SQ.png"
try:
    # Download and load the pipeline and model
    pipeline_path = download_file(pipeline_url, "Vectorizer.pkl")
    model_path = download_file(model_url, "Model.pkl")

    vectorizer = joblib.load(pipeline_path)
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

# Display the image
st.image(image_url, caption='Super Email Spam Detector App', width=500)

# Streamlit app layout
st.title("Welcome to Super Email Spam Detector App (MIS542)")

input_text = st.text_area("Let's Validate Your Email:", "")

if st.button("Let's Check!"):
    if input_text:
        try:
            processed_text = vectorizer.transform([input_text])
            prediction = model.predict(processed_text)
            #result = "Spam" if prediction[0] == 1 else "Not Spam"
            st.write(f"Prediction: {prediction}")
        except Exception as e:
            st.error(f"Error during Classification: {e}")
    else:
        st.write("Please enter a message to check.")

# Display the image
st.image(image_url2, caption='Super Email Spam Detector App', use_column_width=True)
