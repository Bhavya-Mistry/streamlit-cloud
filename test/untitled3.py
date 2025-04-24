import streamlit as st
import pickle
import numpy as np
import os

# Load the scaler and model using paths relative to this file
scaler_path = os.path.join(os.path.dirname(__file__), 'scaler.pkl')
model_path = os.path.join(os.path.dirname(__file__), 'kmeans_model.pkl')

with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

with open(model_path, 'rb') as f:
    kmeans = pickle.load(f)

# App title
st.title("🛍️ Customer Segmentation using K-Means")

st.write("""
This mini tool predicts which customer segment a user belongs to, based on their **Recency**, **Frequency**, and **Monetary** values.
""")

# Input fields
recency = st.number_input("Recency (days since last purchase)", min_value=0, step=1)
frequency = st.number_input("Frequency (number of purchases)", min_value=0, step=1)
monetary = st.number_input("Monetary (total spend)", min_value=0.0, step=1.0, format="%.2f")

# Predict button
if st.button("Predict Segment"):
    # Prepare input
    input_data = np.array([[recency, frequency, monetary]])
    
    # Scale input
    scaled_input = scaler.transform(input_data)
    
    # Predict cluster
    cluster = kmeans.predict(scaled_input)[0]
    
    # Interpretation (optional)
    segments = {
        0: "Loyal Customers",
        1: "Churned/Inactive Customers",
        2: "High Value Customers",
        3: "New or Low Engagement Customers"
    }
    
    st.success(f"🔍 Predicted Segment: **Cluster {cluster} - {segments.get(cluster, 'Unknown')}**")
