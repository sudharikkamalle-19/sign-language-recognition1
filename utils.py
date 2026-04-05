import cv2
import numpy as np

def preprocess_frame(frame):
    # Resize
    frame = cv2.resize(frame, (64, 64))

    # Normalize
    frame = frame / 255.0

    # Expand dimensions for model
    frame = np.expand_dims(frame, axis=0)

    return frame