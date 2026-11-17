"""
Predict one image.
Run:  python predict.py path/to/eye_image.png
"""
import sys
import numpy as np
import tensorflow as tf
from PIL import Image
from config import MODEL_PATH, IMG_SIZE, CLASS_NAMES

_model = None


def get_model():
    global _model
    if _model is None:
        _model = tf.keras.models.load_model(MODEL_PATH)
    return _model


def predict_image(path):
    img = Image.open(path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    arr = np.expand_dims(np.array(img, dtype="float32"), axis=0)  # shape (1,224,224,3)
    probs = get_model().predict(arr, verbose=0)[0]
    best = int(np.argmax(probs))

    all_probs = []
    for i in range(len(CLASS_NAMES)):
        all_probs.append({"label": CLASS_NAMES[i], "probability": round(float(probs[i]) * 100, 2)})

    return {"grade": best, "label": CLASS_NAMES[best],
            "confidence": round(float(probs[best]) * 100, 2), "all": all_probs}


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict.py <image_path>")
    else:
        print(predict_image(sys.argv[1]))
