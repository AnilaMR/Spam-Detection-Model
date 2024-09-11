import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your trained model and TF-IDF vectorizer
Model = joblib.load('spam_classifier_model.pkl')
vectorizer = joblib.load('tfidf_vectorizer.pkl')

# App Title and Description
st.title("🚫 Spam Email Classifier")
st.markdown("""
    **Welcome to the Spam Email Classifier!**  
    Enter the content of an email below, and this tool will predict whether it's spam or not spam.  
    This app uses a machine learning model trained on a spam email dataset. 🎯  
    """)
st.write("")

# Sidebar for additional information
st.sidebar.title("About")
st.sidebar.info("""
    This app uses a Support Vector Machine model trained on email data to predict spam.  
    You can enter any email text in the box, and the model will classify it as spam or not spam.  
    The app is built using Streamlit.  
""")

# Input Section
st.header("Enter Email Content:")
email_input = st.text_area("Copy and paste the email content here:")

# Prediction Function
def predict_spam(text):
    # Transform text using the loaded TF-IDF vectorizer
    text_transformed = vectorizer.transform([text])
    prediction = Model.predict(text_transformed)
    return prediction[0]

# Prediction Button and Result
if st.button("Classify Email"):
    if email_input:
        prediction = predict_spam(email_input)
        if prediction == 1:
            st.error("⚠️ This email is classified as **SPAM**.")
        else:
            st.success("✅ This email is classified as **NOT SPAM**.")
    else:
        st.warning("Please enter some email content to classify.")

# Footer with styling
st.markdown("""
    <style>
    footer {
        visibility: hidden;
    }
    footer:after {
        content:'Created with ❤️ by Anila';
        visibility: visible;
        display: block;
        position: relative;
        padding: 5px;
        top: 2px;
    }
    </style>
""", unsafe_allow_html=True)











        

       
    
