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

model = tf.keras.models.load_model("/home/project/app/resnet50_cancer_classifier.keras")

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
        brightness_max=144,
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
st.set_page_config(page_title="Cancer Classifier", page_icon=":microscope:", layout="centered")
st.title(":microscope: Cancer Tissue Classifier")
st.caption("ResNet50 Breast Control Prostate Skin")

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
