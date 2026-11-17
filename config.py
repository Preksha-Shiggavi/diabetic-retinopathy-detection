import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Image + training settings
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS_HEAD = 8    # train only the new top layers
EPOCHS_FINE = 6    # then fine-tune the last layers of MobileNetV2
FINE_TUNE_LAST = 30  # how many base layers to unfreeze

# Folders
RAW_DIR = os.path.join(BASE_DIR, "dataset", "raw")
TRAIN_DIR = os.path.join(BASE_DIR, "dataset", "train")
VAL_DIR = os.path.join(BASE_DIR, "dataset", "val")
MODEL_PATH = os.path.join(BASE_DIR, "models", "dr_model.keras")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")

# Folder names are 0..4 (APTOS labels); these are the readable names
FOLDER_NAMES = ["0", "1", "2", "3", "4"]
CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
