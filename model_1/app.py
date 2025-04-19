import streamlit as st

# Load your model
import pickle
import os

# Use the correct relative path from streamlit_app.py to the .pkl file
model_path = os.path.join(os.path.dirname(__file__), 'salary_predictor.pkl')

with open(model_path, 'rb') as file:
    model = pickle.load(file)


# Your Streamlit code here
st.title("My ML Model")
input1 = st.number_input("Feature 1")

input_data = [[input1]]
prediction = model.predict(input_data)
st.write(f"Prediction: {prediction}")
