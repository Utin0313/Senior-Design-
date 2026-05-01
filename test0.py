#!/usr/bin/env python
import streamlit as st
import tensorflow as tf
import cv2
import atexit
import time
import numpy as np
from PIL import Image
from picamera2 import Picamera2

# -------------------------
# MODEL
# -------------------------
CLASS_NAMES = ["Breast", "Control", "Prostate", "Skin"]

model = tf.keras.models.load_model(
    "/home/project/app/resnet50_cancer_classifier.keras"
)

# -------------------------
# CAMERA (singleton)
# -------------------------
if st.session_state.get("picam2") is None:
    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration())
    time.sleep(2)
    picam2.start()
    st.session_state.picam2 = picam2
else:
    picam2 = st.session_state.picam2


def cleanup():
    cam = st.session_state.get("picam2")
    if cam is not None:
        try:
            cam.stop()
            cam.close()
        except:
            pass


atexit.register(cleanup)


def capture_frame():
    frame = picam2.capture_array()

    # Ensure uint8 RGB
    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    return frame


# -------------------------
# CROPPING SETTINGS
# -------------------------
CROP_X, CROP_Y, CROP_W, CROP_H = 1740, 1032, 402, 762
CAM_X_PIXEL, CAM_Y_PIXEL = 4056, 3040


# -------------------------
# PREPROCESSING PIPELINE
# -------------------------
def preprocess(frame):

    # --- Convert to PIL RGB ---
    img = Image.fromarray(frame).convert("RGB")
    width, height = img.size

    # --- Normalize crop ---
    x1 = CROP_X / CAM_X_PIXEL
    x2 = (CROP_X + CROP_W) / CAM_X_PIXEL
    y1 = CROP_Y / CAM_Y_PIXEL
    y2 = (CROP_Y + CROP_H) / CAM_Y_PIXEL

    left = int(x1 * width)
    right = int(x2 * width)
    top = int(y1 * height)
    bottom = int(y2 * height)

    img = img.crop((left, top, right, bottom))

    st.image(img, caption="1. Cropped Input")

    # --- Convert to numpy RGB ---
    img_np = np.array(img).astype(np.float32)

    # -------------------------
    # STRIP DETECTION MASK
    # -------------------------
    hsv = cv2.cvtColor(img_np.astype(np.uint8), cv2.COLOR_RGB2HSV)

    saturation = hsv[:, :, 1]

    # test strip = high saturation
    strip_mask = (saturation > 80).astype(np.float32)

    strip_mask_3ch = np.repeat(strip_mask[:, :, None], 3, axis=2)

    # -------------------------
    # BLACK BACKGROUND
    # -------------------------
    img_np = img_np * strip_mask_3ch

    st.image(img_np.astype(np.uint8), caption="2. After Black Background Mask")

    # -------------------------
    # NOISE ONLY ON STRIP
    # -------------------------
    noise = np.random.normal(0, 0.05, img_np.shape).astype(np.float32)
    img_np = img_np + noise * strip_mask_3ch

    img_np = np.clip(img_np, 0, 255)

    st.image(img_np.astype(np.uint8), caption="3. After Noise Injection")

    # -------------------------
    # RESIZE
    # -------------------------
    img_np = cv2.resize(img_np, (224, 224))

    st.image(img_np.astype(np.uint8), caption="4. Resized (224x224)")

    # -------------------------
    # RESNET PREPROCESS
    # -------------------------
    img_np = tf.keras.applications.resnet50.preprocess_input(img_np)

    img_np = np.expand_dims(img_np, axis=0)

    return img_np


# -------------------------
# PREDICTION
# -------------------------
def predict(x):
    out = model.predict(x, verbose=0)
    return out[0]


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Cancer Classifier", layout="centered")
st.title(":microscope: Cancer Tissue Classifier")

if st.button("Capture & Classify", use_container_width=True):

    frame = capture_frame()

    st.image(frame, caption="Raw Camera Frame")

    tensor = preprocess(frame)
    probs = predict(tensor)

    top_idx = np.argmax(probs)
    label = CLASS_NAMES[top_idx]
    conf = probs[top_idx] * 100

    col1, col2 = st.columns(2)

    with col1:
        st.image(frame, caption="Input Frame")

    with col2:
        st.metric("Prediction", label)
        st.metric("Confidence", f"{conf:.2f}%")

    st.divider()
    st.subheader("Class probabilities")

    for name, p in zip(CLASS_NAMES, probs):
        st.write(f"{name}: {p*100:.2f}%")
        st.progress(float(p))
