"""
Step 3: Check how good the model is (accuracy, precision, recall, confusion matrix).
Run:  python evaluate.py
"""
import os
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, cohen_kappa_score
from config import *


def main():
    model = tf.keras.models.load_model(MODEL_PATH)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR, labels="inferred", label_mode="categorical",
        class_names=FOLDER_NAMES, image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=False)

    y_true, y_pred = [], []
    for images, labels in val_ds:
        probs = model.predict(images, verbose=0)
        batch_true = np.argmax(labels.numpy(), axis=1)
        batch_pred = np.argmax(probs, axis=1)
        for i in range(len(batch_true)):
            y_true.append(int(batch_true[i]))
            y_pred.append(int(batch_pred[i]))

    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, zero_division=0))
    print("Quadratic Weighted Kappa:", cohen_kappa_score(y_true, y_pred, weights="quadratic"))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    plt.yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    for r in range(cm.shape[0]):
        for c in range(cm.shape[1]):
            plt.text(c, r, str(cm[r, c]), ha="center", va="center")
    plt.xlabel("Predicted"); plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "confusion_matrix.png"))
    print("Saved confusion_matrix.png")


if __name__ == "__main__":
    main()
