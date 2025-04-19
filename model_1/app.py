import streamlit as st
import pickle
import os

# Load the model
model_path = os.path.join(os.path.dirname(__file__), 'salary_predictor.pkl')
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Page config
st.set_page_config(page_title="Salary Predictor", page_icon="💼", layout="centered")

# Custom CSS for styling
st.markdown("""
    <style>
        .main {
            background-color: #f8f9fa;
        }
        .stButton button {
            background-color: #4CAF50;
            color: white;
            font-weight: bold;
            border-radius: 8px;
            padding: 10px 20px;
        }
        .stTextInput input {
            border-radius: 10px;
        }
    </style>
""", unsafe_allow_html=True)

# App title
st.title("💼 Simple Salary Predictor")

# Description
st.markdown("""
Enter a value for the feature below to get a predicted salary.
This model uses a basic linear regression under the hood.
""")

# Input
input1 = st.number_input("📈 Experience", min_value=0.0, step=0.1)

# Predict button
if st.button("🔍 Predict Salary"):
    input_data = [[input1]]
    prediction = model.predict(input_data)
    st.success(f"💰 Predicted Salary: ₹{int(prediction[0])}")

