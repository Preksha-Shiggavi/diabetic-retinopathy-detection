"""
Step 2: Train the model (transfer learning with MobileNetV2).
Run:  python train.py
"""
import os
import tensorflow as tf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tensorflow.keras import layers, models
from config import *


def load_data():
    train_ds = tf.keras.utils.image_dataset_from_directory(
        TRAIN_DIR, labels="inferred", label_mode="categorical",
        class_names=FOLDER_NAMES, image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=True, seed=42)
    val_ds = tf.keras.utils.image_dataset_from_directory(
        VAL_DIR, labels="inferred", label_mode="categorical",
        class_names=FOLDER_NAMES, image_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE, shuffle=False)
    return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)


def get_class_weights():
    """Rare classes (like Severe) get a bigger weight so the model does not ignore them."""
    counts = []
    for i in range(len(FOLDER_NAMES)):
        counts.append(len(os.listdir(os.path.join(TRAIN_DIR, FOLDER_NAMES[i]))))
    total = sum(counts)
    weights = {}
    for i in range(len(counts)):
        weights[i] = total / (len(counts) * counts[i])
    print("Class weights:", weights)
    return weights


def build_model():
    base = tf.keras.applications.MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    base.trainable = False

    inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
    x = layers.RandomFlip("horizontal_and_vertical")(inputs)   # augmentation (training only)
    x = layers.RandomRotation(0.1)(x)
    x = layers.RandomZoom(0.1)(x)
    x = layers.Rescaling(1.0 / 127.5, offset=-1)(x)            # pixels -> range -1..1
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(len(CLASS_NAMES), activation="softmax")(x)
    return models.Model(inputs, outputs), base


def plot_history(histories):
    acc, val_acc, loss, val_loss = [], [], [], []
    for i in range(len(histories)):
        acc += histories[i].history["accuracy"]
        val_acc += histories[i].history["val_accuracy"]
        loss += histories[i].history["loss"]
        val_loss += histories[i].history["val_loss"]
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(acc, label="train"); plt.plot(val_acc, label="val")
    plt.title("Accuracy"); plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(loss, label="train"); plt.plot(val_loss, label="val")
    plt.title("Loss"); plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "training_curves.png"))
    print("Saved training_curves.png")


def main():
    train_ds, val_ds = load_data()
    class_weights = get_class_weights()
    model, base = build_model()

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy",
                                           save_best_only=True, verbose=1),
        tf.keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=4,
                                         restore_best_weights=True),
        tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
    ]

    # Phase 1: train only the new top layers
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    h1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD,
                   class_weight=class_weights, callbacks=callbacks)

    # Phase 2: unfreeze the last few base layers and train gently
    base.trainable = True
    for i in range(len(base.layers) - FINE_TUNE_LAST):
        base.layers[i].trainable = False
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
                  loss="categorical_crossentropy", metrics=["accuracy"])
    h2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FINE,
                   class_weight=class_weights, callbacks=callbacks)

    plot_history([h1, h2])
    print("Best model saved at:", MODEL_PATH)


if __name__ == "__main__":
    main()
