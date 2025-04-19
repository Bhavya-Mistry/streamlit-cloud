import streamlit as st
import joblib
import numpy as np

# Load the saved model
model = joblib.load('salary_predictor.pkl')

# Title of the web app
st.title('Salary Prediction App')

# Input field for the user to enter years of experience
experience = st.number_input('Enter number of years of experience', min_value=0, max_value=50, value=5)

# Predict the salary based on the input
predicted_salary = model.predict(np.array([[experience]]))

# Display the prediction result
salary = float(predicted_salary)
st.write(f'Predicted Salary: ${salary:,.2f}')

