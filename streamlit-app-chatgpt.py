import streamlit as st
from transformers import pipeline

# Set the title of the application
st.title("A sentiment analyser written by ChatGPT")

# Create the input text field
text = st.text_input("Enter some text to analyse:")

# Use the Hugging Face Pipeline API to create a sentiment classifier
sentiment_classifier = pipeline("text-classification", model="distilbert-base-uncased-finetuned-sst-2-english")

# Evaluate the text entered by the user and display the result
if text:
    result = sentiment_classifier(text)[0]
    st.write(f"Predicted sentiment: {result['label']}")