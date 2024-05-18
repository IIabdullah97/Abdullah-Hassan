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
pipeline_url = "https://raw.githubusercontent.com/BaderIAlharbi/Team4/main/Team4vectorizer_pipeline.pkl"
model_url = "https://raw.githubusercontent.com/BaderIAlharbi/Team4/main/Team4model_nb.pkl"
image_url = "https://miro.medium.com/v2/resize:fit:1400/0*mbFBPcPUJD-53v3h.png" 

try:
    # Download and load the pipeline and model
    pipeline_path = download_file(pipeline_url, "Team4vectorizer_pipeline.pkl")
    model_path = download_file(model_url, "Team4model_nb.pkl")

    vectorizer = joblib.load(pipeline_path)
    model = joblib.load(model_path)
except Exception as e:
    st.error(f"Error loading model or vectorizer: {e}")
    st.stop()

# Display the image
st.image(image_url, caption='Team 4 Project', use_column_width=True)

# Streamlit app layout
st.title("Team 4 Project")
st.markdown("### Email Spam Detection App")

input_text = st.text_area("Enter your email text and we will check it for you for free!:", "")

if st.button("Predict"):
    if input_text:
        try:
            processed_text = vectorizer.transform([input_text])
            prediction = model.predict(processed_text)
            #result = "Spam" if prediction[0] == 1 else "Not Spam"
            st.write(f"Prediction: {prediction}")
        except Exception as e:
            st.error(f"Error during prediction: {e}")
    else:
        st.write("Please enter a message to predict.")
