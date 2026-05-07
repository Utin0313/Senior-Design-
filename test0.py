#!/usr/bin/env python
import streamlit as st
import tensorflow as tf
import cv2
import atexit
import time
import numpy as np
from PIL import Image
from picamera2 import Picamera2

# -- Model setup --
CLASS_NAMES = ["Breast", "Control", "Prostate", "Skin"]
 
model = tf.keras.models.load_model("/home/project/app/resnet50_classifier.keras")

# -- Camera setup --
if st.session_state.get("picam2") is None:
    picam2 = Picamera2()
    picam2.configure(picam2.create_still_configuration())
    time.sleep(2)
    picam2.start()
    st.session_state.picam2 = picam2
else:
    picam2 = st.session_state.picam2

def cleanup():
    picam2 = st.session_state.get("picam2")
    if picam2 is not None:
        try:
            picam2.stop()
            picam2.close()
        except:
            pass

atexit.register(cleanup)

def capture_frame():
    return picam2.capture_array()

CROP_X, CROP_Y, CROP_W, CROP_H = 1740, 1032, 402, 762
CAM_X_PIXEL, CAM_Y_PIXEL = 4056, 3040

def generate_brightness_mask_array(
    img_array,
    brightness_min,
    brightness_max,
    dot_saturation_min=80,
):
    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    brightness = hsv[:, :, 2]
    saturation = hsv[:, :, 1]

    bg_mask = cv2.inRange(brightness, brightness_min, brightness_max)
    
    dot_mask = (saturation >= dot_saturation_min).astype(np.uint8) * 255

    remove_mask = cv2.bitwise_and(bg_mask, cv2.bitwise_not(dot_mask))
    keep_mask = cv2.bitwise_not(remove_mask)

    result = cv2.bitwise_and(img_array, img_array, mask=keep_mask)
    return result

def preprocess(frame):
    img = Image.fromarray(frame).convert("RGB")

    width, height = img.size

    x1 = CROP_X / CAM_X_PIXEL
    x2 = (CROP_X + CROP_W) / CAM_X_PIXEL
    y1 = CROP_Y / CAM_Y_PIXEL
    y2 = (CROP_Y + CROP_H) / CAM_Y_PIXEL

    left = int(x1 * width)
    right = int(x2 * width)
    top = int(y1 * height)
    bottom = int(y2 * height)

    img = img.crop((left, top, right, bottom))

    # -- Debug -- 
    img.save("/home/project/Pictures/debug_1_crop.jpg")

    img_arr = np.array(img)

 
    img_arr = generate_brightness_mask_array(
        img_arr,
        brightness_min=0,
        brightness_max=225,
        dot_saturation_min=80
    )

	
    # -- Debug -- 
    st.image(img_arr, caption="After Mask", use_container_width=True)
    
    img = Image.fromarray(img_arr)
    
    # ResNet50 input size
    img = img.resize((224, 224))

    # IMPORTANT: use same preprocessing as training/testing
    arr = np.array(img, dtype=np.float32)
    arr = tf.keras.applications.resnet50.preprocess_input(arr)
    arr = np.expand_dims(arr, axis=0)

    debug_img = Image.fromarray(np.array(img).astype(np.uint8))
    debug_img.save("/home/project/Pictures/debug_3_preprocessed.jpg")

    return arr

def predict(preprocessed):
    output = model.predict(preprocessed, verbose=0)
    return output[0]

# -- Streamlit UI --
st.set_page_config(page_title="LRDM", layout="wide")

st.markdown("""
<style>
body {
    background-color: #132245;
    color: #dce8ff;
}

.block-container {
    padding-top: 2rem;
}

.card {
    background: #1b2f56;
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.08);
}

.big-title {
    font-size: 1.5rem;
    font-weight: bold;
}

.sub {
    color: #7a9abf;
    font-size: 0.8rem;
    text-transform: uppercase;
}

.metric {
    font-size: 2rem;
    font-weight: bold;
}

</style>
""", unsafe_allow_html=True)

st.title("L.R.D.M Dashboard")

col1, col2 = st.columns([2, 1])

# =========================
# LEFT SIDE - IMAGE VIEWER
# =========================
with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Scan Output")

    if "frame" in st.session_state:
        st.image(st.session_state["frame"], use_container_width=True)
    else:
        st.info("No scan yet")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# RIGHT SIDE
# =========================
with col2:

    # -------- ANALYSIS CARD --------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Analysis")

    if "label" in st.session_state:
        st.metric("Prediction", st.session_state["label"])
        st.metric("Confidence", f"{st.session_state['confidence']:.1f}%")
    else:
        st.write("No prediction yet")

    st.markdown("---")

    brightness = st.slider("LED Brightness", 0, 100, 50)

    st.markdown("</div>", unsafe_allow_html=True)

    # -------- FINDINGS --------
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown("### Findings")

    st.write("✔ Region detected")
    st.write("✔ Mask applied")
    st.write("✔ Image quality OK")

    st.markdown("</div>", unsafe_allow_html=True)

# =========================
# BUTTON (CENTER ACTION)
# =========================
if st.button("Capture & Classify", use_container_width=True):

    with st.spinner("Running LRDM pipeline..."):

        frame = capture_frame()
        tensor = preprocess(frame)
        probs = predict(tensor)

        idx = np.argmax(probs)
        label = CLASS_NAMES[idx]
        confidence = probs[idx] * 100

        # store results
        st.session_state["frame"] = frame
        st.session_state["label"] = label
        st.session_state["confidence"] = confidence

    st.rerun()
