# Lateral Rapid Diagnostic Machine (L.R.D.M.)

## Overview
The Lateral Rapid Diagnostic Machine (L.R.D.M.) is a portable, AI-enabled diagnostic system designed to automate and enhance lateral flow assay (LFA) testing for rapid disease detection.

This project integrates machine learning, embedded systems, and hardware design to create a low-cost, real-time diagnostic device capable of analyzing test strips and producing accurate results without manual interpretation.

---

## Motivation
Rapid diagnostic tests are widely used for detecting infectious diseases due to their affordability and ease of use. However, traditional test interpretation is manual, subjective, and prone to error.

L.R.D.M. addresses these limitations by:
- Automating result interpretation using computer vision
- Reducing human error in diagnosis
- Enabling consistent and repeatable results
- Providing a low-cost alternative to expensive commercial systems

---

## System Architecture
The system combines hardware and machine learning components:

### Hardware
- Imaging system for capturing LFA strip results
- Sensors and PCB integration for device control
- Portable enclosure designed using CAD

### Software / AI
- Image preprocessing pipeline
- CNN-based classification model for test result detection
- Real-time inference for rapid diagnostics

---

## Features
- Real-time detection and classification of test results  
- Automated image-based analysis of lateral flow assays  
- Low-cost and portable design for point-of-care (POC) use  
- Scalable architecture for detecting multiple disease types  
- Integration of AI with embedded hardware systems  

---

## Machine Learning Pipeline
1. Image Acquisition  
2. Preprocessing (normalization, resizing, noise handling)  
3. Feature Extraction using Convolutional Neural Networks  
4. Classification of test results (e.g., positive / negative)  
5. Output display and decision support  

---

## Applications
- Infectious disease detection (e.g., COVID-19, malaria, etc.)  
- Point-of-care diagnostics in low-resource environments  
- Medical screening and rapid testing facilities  
- Field diagnostics and remote healthcare  

---

## Tech Stack
- Python  
- TensorFlow / Keras  
- Embedded Systems & Sensors  
- PCB Design  
- CAD Modeling  

---

## Team
- Austin Thanh Trinh  
- Josh Vo  
- Jason Yao  
- Rebaz Kamal  
- Kamran Agayev  
- Mohammed Marafih  
- Josselyn Mata Calidonio  
- Kimberly Hamad-Schifferli  

---

## Future Work
- Improve model accuracy and robustness across varying lighting conditions  
- Expand classification to multi-class disease detection  
- Optimize model for edge deployment on embedded hardware  
- Integrate cloud-based monitoring and data tracking  

---

## Impact
This project aims to make rapid diagnostic testing:
- More accessible  
- More reliable  
- More affordable  

By combining AI and hardware, L.R.D.M. has the potential to improve healthcare delivery, especially in underserved and resource-limited environments.

---
