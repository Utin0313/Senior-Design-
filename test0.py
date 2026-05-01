#!/usr/bin/env python
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
from picamera2 import Picamera2
import time

# -------------------------
# MODEL
# -------------------------
CLASS_NAMES = ["Breast", "Control", "Prostate", "Skin"]

model = tf.keras.models.load_model(
    "/home/project/app/resnet50_cancer_classifier.keras"
)

# -------------------------
# CAMERA SETUP
# -------------------------
if st.session_state.get("picam2") is None:
    picam2 = Picamera2()

    picam2.configure(picam2.create_still_configuration())

    picam2.start()
    time.sleep(2)

    # 🔥 LOCK CAMERA (critical for stability)
    picam2.set_controls({
        "AeEnable": False,   # disable auto exposure
        "AwbEnable": False,  # disable auto white balance
    })

    st.session_state.picam2 = picam2
else:
    picam2 = st.session_state.picam2


# -------------------------
# CAPTURE FRAME
# -------------------------
def capture_frame():
    frame = picam2.capture_array()

    if frame.dtype != np.uint8:
        frame = frame.astype(np.uint8)

    return frame


# -------------------------
# CROP SETTINGS
# -------------------------
CROP_X, CROP_Y, CROP_W, CROP_H = 1740, 1032, 402, 762
CAM_X_PIXEL, CAM_Y_PIXEL = 4056, 3040


# -------------------------
# PREPROCESS (REAL-TIME ONLY FIX)
# -------------------------
def preprocess(frame):

    img = Image.fromarray(frame).convert("RGB")
    width, height = img.size

    # --- fixed crop ---
    x1 = int((CROP_X / CAM_X_PIXEL) * width)
    x2 = int(((CROP_X + CROP_W) / CAM_X_PIXEL) * width)
    y1 = int((CROP_Y / CAM_Y_PIXEL) * height)
    y2 = int(((CROP_Y + CROP_H) / CAM_Y_PIXEL) * height)

    img = img.crop((x1, y1, x2, y2))

    st.image(img, caption="Cropped Input (Model sees this)")

    # --- mild normalization (camera drift correction only) ---
    img_np = np.array(img).astype(np.float32)

    mean = np.mean(img_np)
    img_np = (img_np / (mean + 1e-6)) * 128.0
    img_np = np.clip(img_np, 0, 255)

    # --- resize (PIL for stability) ---
    img = Image.fromarray(img_np.astype(np.uint8))
    img = img.resize((224, 224), Image.BICUBIC)

    img_np = np.array(img).astype(np.float32)

    # --- ResNet preprocessing (must match training) ---
    img_np = tf.keras.applications.resnet50.preprocess_input(img_np)

    return np.expand_dims(img_np, axis=0)


# -------------------------
# PREDICTION
# -------------------------
def predict(x):
    return model.predict(x, verbose=0)[0]


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Cancer Classifier", layout="centered")
st.title(":microscope: Cancer Tissue Classifier (Stable Inference Mode)")

<<<<<<< HEAD
uploaded = st.file_uploader("/home/project/Documents/Senior-Design-/Breast_1_Test_1_20251216_143828_masked", type=["png", "jpg", "jpeg"])
=======
st.caption("Real-time inference stabilized (no retraining required)")
>>>>>>> 42b57f65b0123422605c3668f47c4cede7ea1900

if st.button("Capture & Classify", use_container_width=True):

    # capture
    frame = capture_frame()

    st.image(frame, caption="Raw Camera Frame")

    # preprocess
    tensor = preprocess(frame)

    # predict
    probs = predict(tensor)

    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx]
    confidence = probs[idx] * 100

    # results
    col1, col2 = st.columns(2)

    with col1:
        st.metric("Prediction", label)
        st.metric("Confidence", f"{confidence:.2f}%")

    with col2:
        st.image(frame, caption="Input Frame")

    st.divider()
    st.subheader("Class Probabilities")

    for name, p in zip(CLASS_NAMES, probs):
        st.write(f"{name}: {p*100:.2f}%")
        st.progress(float(p))

    st.divider()
    st.write("Raw output:", probs)
