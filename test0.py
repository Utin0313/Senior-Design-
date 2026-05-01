#!/usr/bin/env python
import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import cv2

# -------------------------
# MODEL
# -------------------------
CLASS_NAMES = ["Breast", "Control", "Prostate", "Skin"]

model = tf.keras.models.load_model(
    "/home/project/app/resnet50_cancer_classifier.keras"
)

# -------------------------
# SAME PREPROCESSING YOU USE IN DEPLOYMENT
# -------------------------
def preprocess(img_np):

    # ensure RGB
    if isinstance(img_np, Image.Image):
        img_np = np.array(img_np.convert("RGB"))

    # resize EXACTLY like inference (IMPORTANT)
    img = Image.fromarray(img_np.astype(np.uint8))
    img = img.resize((224, 224), Image.BICUBIC)

    img_np = np.array(img).astype(np.float32)

    # IMPORTANT: match training expectation
    img_np = tf.keras.applications.resnet50.preprocess_input(img_np)

    return np.expand_dims(img_np, axis=0)


def predict(x):
    return model.predict(x, verbose=0)[0]


# -------------------------
# UI
# -------------------------
st.set_page_config(page_title="Validation Test", layout="centered")
st.title("🧪 Validation Image Model Test")

uploaded = st.file_uploader("Upload a validation image", type=["png", "jpg", "jpeg"])

if uploaded is not None:

    # load image
    img = Image.open(uploaded).convert("RGB")

    st.image(img, caption="Validation Image (Input)")

    # preprocess
    tensor = preprocess(img)

    # predict
    probs = predict(tensor)

    # results
    idx = int(np.argmax(probs))
    label = CLASS_NAMES[idx]
    conf = probs[idx] * 100

    st.subheader("Prediction Result")
    st.metric("Class", label)
    st.metric("Confidence", f"{conf:.2f}%")

    st.divider()
    st.subheader("Probabilities")

    for name, p in zip(CLASS_NAMES, probs):
        st.write(f"{name}: {p*100:.2f}%")
        st.progress(float(p))

    st.divider()
    st.write("Raw output:", probs)
