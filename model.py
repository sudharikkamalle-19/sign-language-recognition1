import numpy as np

# Dummy labels (you can extend)
LABELS = ["HI", "BYE", "SUPER", "Hello", "Thanks"]

class DummyModel:
    def predict(self, x):
        return np.random.rand(1, len(LABELS))

def load_model():
    # Replace with real model loading later
    return DummyModel()

def predict_gesture(model, frame):
    preds = model.predict(frame)
    index = np.argmax(preds)
    return LABELS[index]