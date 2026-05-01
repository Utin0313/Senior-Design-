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

    img = Image.fromarray(frame).convert("RGB")

    # --- crop ---
    width, height = img.size

    x1 = CROP_X / CAM_X_PIXEL
    x2 = (CROP_X + CROP_W) / CAM_X_PIXEL
    y1 = CROP_Y / CAM_Y_PIXEL
    y2 = (CROP_Y + CROP_H) / CAM_Y_PIXEL

    img = img.crop((
        int(x1 * width),
        int(y1 * height),
        int(x2 * width),
        int(y2 * height)
    ))

    st.image(img, caption="Cropped Input")

    img_np = np.array(img).astype(np.uint8)

    # --- LIGHT contrast enhancement (NOT masking) ---
    lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    img_np = cv2.merge((l, a, b))
    img_np = cv2.cvtColor(img_np, cv2.COLOR_LAB2RGB)

    st.image(img_np, caption="Contrast Enhanced (CLAHE)")

    # --- resize ---
    img_np = cv2.resize(img_np, (224, 224))

    # --- ResNet preprocessing ---
    img_np = tf.keras.applications.resnet50.preprocess_input(
        img_np.astype(np.float32)
    )

    return np.expand_dims(img_np, axis=0)


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
