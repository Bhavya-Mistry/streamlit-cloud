# dehaze_app.py
import streamlit as st
import cv2
import numpy as np
from PIL import Image

# DCP Functions (use the same ones you implemented earlier)
def get_dark_channel(image, size=15):
    min_img = cv2.min(cv2.min(image[:,:,0], image[:,:,1]), image[:,:,2])
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (size, size))
    dark_channel = cv2.erode(min_img, kernel)
    return dark_channel

def get_atmosphere(image, dark_channel):
    h, w = image.shape[:2]
    num_pixels = h * w
    num_brightest = int(max(num_pixels * 0.001, 1))
    flat_dark = dark_channel.flatten()
    indices = np.argsort(flat_dark)[-num_brightest:]
    brightest = image.reshape(num_pixels, 3)[indices]
    A = np.max(brightest, axis=0)
    return A

def get_transmission(image, A, omega=0.95, size=15):
    normed = image / A
    dark_channel = get_dark_channel(normed, size)
    transmission = 1 - omega * dark_channel
    return transmission

def recover_scene_radiance(image, transmission, A, t0=0.1):
    transmission = np.clip(transmission, t0, 1.0)
    recovered = np.empty_like(image, dtype=np.float32)
    for i in range(3):
        recovered[:,:,i] = (image[:,:,i] - A[i]) / transmission + A[i]
    recovered = np.clip(recovered, 0, 255).astype(np.uint8)
    return recovered

# Streamlit UI
st.title("Single Image Dehazing using Dark Channel Prior")
uploaded_file = st.file_uploader("Upload a hazy image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    dark_channel = get_dark_channel(image_np)
    A = get_atmosphere(image_np, dark_channel)
    transmission = get_transmission(image_np, A)
    dehazed = recover_scene_radiance(image_np, transmission, A)

    st.subheader("Dehazed Image")
    st.image(cv2.cvtColor(dehazed, cv2.COLOR_BGR2RGB), use_column_width=True)
