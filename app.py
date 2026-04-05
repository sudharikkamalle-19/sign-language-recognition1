import streamlit as st
import cv2
import numpy as np
from model import load_model, predict_gesture
from utils import preprocess_frame
import time

st.set_page_config(page_title="Sign Language Recognition", layout="centered")

st.title("✋ Sign Language Recognition App")

# Load model
model = load_model()

# Start webcam
run = st.checkbox("Start Camera")

FRAME_WINDOW = st.image([])

cap = None


if run:
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera error")
            break

        frame = cv2.flip(frame, 1)

        processed = preprocess_frame(frame)
        prediction = predict_gesture(model, processed)

        cv2.putText(frame, f"Prediction: {prediction}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        FRAME_WINDOW.image(frame, channels="BGR")

        time.sleep(0.03)  # prevents CPU overload

        # STOP BUTTON FIX
        if not st.session_state.get("run", True):
            break

    cap.release()