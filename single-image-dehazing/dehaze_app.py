import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
with open('toxic_comment_model.pkl', 'rb') as f:
    model = pickle.load(f)

# TF-IDF Vectorizer
with open('tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

st.title("Toxic Comment Classifier")
st.write("Enter a comment below:")

user_input = st.text_area("Comment")

if st.button("Classify"):
    if user_input:
        input_vec = vectorizer.transform([user_input])
        prediction = model.predict(input_vec)
        label = "Toxic" if prediction[0] == 1 else "Non-toxic"
        st.success(f"The comment is: **{label}**")
    else:
        st.warning("Please enter some text.")
