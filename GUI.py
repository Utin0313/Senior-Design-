#!/usr/bin/env python
import streamlit as st
import tensorflow as tf
import cv2
import atexit
import time
import os
import numpy as np
import threading
from PIL import Image
from picamera2 import Picamera2
from gpiozero import PWMLED, Button, RotaryEncoder

# --- GPIO Configuration ---
PIN_PWM = 13
PIN_CLK = 4
PIN_DT = 17
PIN_SW = 27

# --- Constants ---
CLASS_NAMES = ["Breast", "Control", "Prostate", "Skin"]
MODEL_PATH = "/home/project/app/resnet50_simple_classifier.keras"
CROP_X, CROP_Y, CROP_W, CROP_H = 1740, 1032, 402, 762
CAM_X_PIXEL, CAM_Y_PIXEL = 4056, 3040

# --- Helper Functions (The missing logic) ---

def generate_brightness_mask_array(img_array, brightness_min, brightness_max, dot_saturation_min=80, strip_brightness_min=140):
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    brightness = hsv[:, :, 2]
    saturation = hsv[:, :, 1]

    bg_mask = cv2.inRange(brightness, brightness_min, brightness_max)
    dot_mask = (saturation >= dot_saturation_min).astype(np.uint8) * 255
    strip_mask = (brightness >= strip_brightness_min).astype(np.uint8) * 255

    keep_mask = cv2.bitwise_or(dot_mask, strip_mask)
    remove_mask = cv2.bitwise_and(bg_mask, cv2.bitwise_not(keep_mask))
    keep_mask_final = cv2.bitwise_not(remove_mask)

    return cv2.bitwise_and(img_array, img_array, mask=keep_mask_final)

def preprocess(frame):
    img = Image.fromarray(frame).convert("RGB")
    width, height = img.size

    left = int((CROP_X / CAM_X_PIXEL) * width)
    right = int(((CROP_X + CROP_W) / CAM_X_PIXEL) * width)
    top = int((CROP_Y / CAM_Y_PIXEL) * height)
    bottom = int(((CROP_Y + CROP_H) / CAM_Y_PIXEL) * height)

    img = img.crop((left, top, right, bottom))
    img_arr = generate_brightness_mask_array(np.array(img), 0, 144, 80)
    
    img = Image.fromarray(img_arr).resize((224, 224))
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.resnet50.preprocess_input(arr)
    return np.expand_dims(arr, axis=0)

# --- Hardware & Model Initialization ---

if "initialized" not in st.session_state:
    # Model
    st.session_state.model = tf.keras.models.load_model(MODEL_PATH)
    
    # Camera
    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration())
    picam2.start()
    st.session_state.picam2 = picam2
    
    # GPIO Components
    st.session_state.led = PWMLED(PIN_PWM, frequency=1000)
    st.session_state.led.value = 0.5
    st.session_state.encoder = RotaryEncoder(PIN_CLK, PIN_DT, wrap=False)
    st.session_state.photo_btn = Button(PIN_SW, pull_up=False, bounce_time=0.2)
    
    st.session_state.initialized = True

# --- Classification Logic ---

def run_classification():
    # Capture and Predict
    frame = st.session_state.picam2.capture_array()
    tensor = preprocess(frame)
    probs = st.session_state.model.predict(tensor, verbose=0)[0]
    
    # Update Session State
    st.session_state.last_prediction = {
        "frame": frame,
        "label": CLASS_NAMES[np.argmax(probs)],
        "confidence": np.max(probs) * 100,
        "probs": probs
    }
    # This dummy variable triggers Streamlit to rerun when changed
    st.session_state.update_trigger = time.time()

# --- Background Hardware Thread ---

def hardware_loop():
    def on_button_press():
        # We must use a separate logic handler because we are in a thread
        run_classification()

    def on_rotate():
        new_val = (st.session_state.encoder.value + 1) / 2
        st.session_state.led.value = max(0.0, min(1.0, new_val))

    st.session_state.photo_btn.when_pressed = on_button_press
    st.session_state.encoder.when_rotated = on_rotate

if "hw_thread_started" not in st.session_state:
    thread = threading.Thread(target=hardware_loop, daemon=True)
    thread.start()
    st.session_state.hw_thread_started = True

# --- Streamlit UI Layout ---

st.set_page_config(page_title="Cancer Classifier", layout="centered")
st.title(":microscope: Cancer Tissue Classifier")

if st.button(":camera: Capture & Classify", use_container_width=True):
    run_classification()

if "last_prediction" in st.session_state:
    res = st.session_state.last_prediction
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(res["frame"], caption="Captured Frame")
    
    with col2:
        st.metric("Prediction", res["label"])
        st.metric("Confidence", f"{res['confidence']:.1f}%")
        
        if res["label"] == "Control":
            st.success("No cancer tissue detected")
        else:
            st.warning(f"{res['label']} cancer tissue detected")

    st.divider()
    for name, prob in zip(CLASS_NAMES, res["probs"]):
        st.progress(float(prob), text=f"{name}: {prob*100:.1f}%")

# Cleanup on exit
def cleanup():
    if "picam2" in st.session_state:
        st.session_state.picam2.stop()
atexit.register(cleanup)
