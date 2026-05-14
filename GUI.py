#!/usr/bin/env python
import streamlit as st
import tensorflow as tf
import cv2
import atexit
import time
import numpy as np
from PIL import Image
from picamera2 import Picamera2
from gpiozero import PWMLED, RotaryEncoder

# --- Hardware Setup (Integrated from breadboard.py) ---
PIN_PWM = 13
PIN_CLK = 4
PIN_DT = 17

# Initialize hardware in session state to prevent re-initialization on rerun
if "led" not in st.session_state:
    st.session_state.led = PWMLED(PIN_PWM, frequency=1000)
    st.session_state.encoder = RotaryEncoder(PIN_CLK, PIN_DT, wrap=False)
    # Set default brightness
    st.session_state.brightness_val = 0.5 
    st.session_state.led.value = 0.5

# --- Model setup ---
CLASS_NAMES = ["Breast", "Control", "Prostate", "Skin"]

@st.cache_resource
def get_model():
    # Utilizing your ResNet50 CNN for disease classification
    return tf.keras.models.load_model("/home/project/app/resnet50_classifier.keras")

model = get_model()

# --- Camera setup ---
@st.cache_resource
def get_camera():
    p2 = Picamera2()
    p2.configure(p2.create_still_configuration())
    time.sleep(2)
    p2.start()
    return p2 

picam2 = get_camera()

def cleanup():
    if "led" in st.session_state:
        st.session_state.led.close()
    if "encoder" in st.session_state:
        st.session_state.encoder.close()
    try:
        picam2.stop()
        picam2.close()
    except:
        pass

atexit.register(cleanup)

# --- Hardware Logic ---
def sync_hardware_to_gui():
    """Update LED brightness based on the slider in the GUI."""
    st.session_state.led.value = st.session_state.brightness_val

def handle_encoder():
    """Read physical encoder and update the session state for the GUI."""
    # Normalize encoder (-1 to 1) to LED/Slider (0.0 to 1.0)
    new_val = (st.session_state.encoder.value + 1) / 2
    clamped_val = max(0.0, min(1.0, new_val))
    if clamped_val != st.session_state.brightness_val:
        st.session_state.brightness_val = clamped_val
        st.session_state.led.value = clamped_val
        st.rerun() # Refresh GUI to move the slider

# Link physical encoder rotation to the handler
st.session_state.encoder.when_rotated = handle_encoder

# --- Image Processing & Prediction ---
# (Keeping your existing capture_frame, generate_brightness_mask_array, 
# and preprocess functions exactly as they were)
def capture_frame():
    return picam2.capture_array()

CROP_X, CROP_Y, CROP_W, CROP_H = 1740, 1032, 402, 762
CAM_X_PIXEL, CAM_Y_PIXEL = 4056, 3040

def preprocess(frame):
    img = Image.fromarray(frame).convert("RGB")
    width, height = img.size
    left = int((CROP_X / CAM_X_PIXEL) * width)
    right = int(((CROP_X + CROP_W) / CAM_X_PIXEL) * width)
    top = int((CROP_Y / CAM_Y_PIXEL) * height)
    bottom = int(((CROP_Y + CROP_H) / CAM_Y_PIXEL) * height)
    img = img.crop((left, top, right, bottom))
    
    img_arr = np.array(img)
    # Custom brightness masking for LFA image capture
    img_arr = generate_brightness_mask_array(img_arr, 0, 225, 80)
    img = Image.fromarray(img_arr).resize((224, 224))
    
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.resnet50.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

def predict(preprocessed):
    output = model.predict(preprocessed, verbose=0)
    return output[0]

# --- Streamlit UI ---
st.set_page_config(page_title="Cancer Classifier", page_icon=":microscope:", layout="centered")
st.title(":microscope: Cancer Tissue Classifier")
st.caption("ResNet50 Breast Control Prostate Skin")

# --- Brightness Tracker & Interactive Slider ---
st.write("### LED Lighting Control")
# This slider is updated by the knob and can also be dragged by the user
st.slider(
    "LED Brightness Level", 
    0.0, 1.0, 
    key="brightness_val", 
    on_change=sync_hardware_to_gui,
    help="Adjust via physical knob or drag this slider"
)
st.metric("Current Intensity", f"{int(st.session_state.brightness_val * 100)}%")

# --- Main Action Button ---
if st.button(":microscope: Capture & Classify", use_container_width=True):
    with st.spinner("Capturing and analysing..."):
        frame = capture_frame()
        tensor = preprocess(frame)
        probs = predict(tensor)

        top_idx = np.argmax(probs)
        label = CLASS_NAMES[top_idx]
        confidence = probs[top_idx] * 100

    col1, col2 = st.columns(2)
    with col1:
        st.image(frame, caption="Captured frame", use_container_width=True)
    with col2:
        st.metric("Prediction", label)
        st.metric("Confidence", f"{confidence:.1f}%")
        if label == "Control":
            st.success("No cancer tissue detected")
        else:
            st.warning(f"{label} cancer tissue detected")

    st.divider()
    st.subheader("All class probabilities")
    for name, prob in zip(CLASS_NAMES, probs):
        st.progress(float(prob), text=f"{name}: {prob*100:.1f}%")
