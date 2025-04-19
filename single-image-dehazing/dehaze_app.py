import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

# Dark Channel Prior functions (reuse your functions here)
def get_dark_channel(image, window_size):
    min_channel = np.min(image, axis=2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (window_size, window_size))
    dark_channel = cv2.erode(min_channel, kernel)
    return dark_channel

def estimate_atmospheric_light(image, dark_channel):
    h, w = dark_channel.shape
    n_pixels = h * w
    n_brightest = int(max(n_pixels * 0.001, 1))
    flat_dark = dark_channel.ravel()
    indices = np.argpartition(flat_dark, -n_brightest)[-n_brightest:]
    brightest = np.unravel_index(indices, dark_channel.shape)
    brightest_pixels = image[brightest]
    A = np.max(brightest_pixels, axis=0)
    return A

def estimate_transmission(image, A, omega=0.95, window_size=15):
    norm_image = image / A
    dark_channel = get_dark_channel(norm_image, window_size)
    transmission = 1 - omega * dark_channel
    return transmission

def recover_scene_radiance(image, A, transmission, t0=0.1):
    transmission = np.clip(transmission, t0, 1)
    J = (image - A) / transmission[:, :, np.newaxis] + A
    J = np.clip(J, 0, 255).astype(np.uint8)
    return J

# Streamlit app
st.title("🌫️ Image Dehazing using Dark Channel Prior")
st.write("Upload a hazy image to dehaze it using the DCP algorithm.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    img_np = np.array(image)

    st.subheader("Original Image")
    st.image(image, use_column_width=True)

    # Resize image for faster processing (optional)
    resized = cv2.resize(img_np, (512, 512))

    # Apply DCP algorithm
    dark = get_dark_channel(resized, window_size=15)
    A = estimate_atmospheric_light(resized, dark)
    transmission = estimate_transmission(resized, A)
    dehazed = recover_scene_radiance(resized, A, transmission)

    st.subheader("Dehazed Image")
    st.image(dehazed, channels="RGB", use_column_width=True)

    # Convert the image to a byte stream for download
    dehazed_image = Image.fromarray(dehazed)
    buffered = BytesIO()
    dehazed_image.save(buffered, format="PNG")
    buffered.seek(0)

    # Option to download
    st.download_button("Download Dehazed Image", data=buffered, file_name="dehazed_image.png", mime="image/png")
