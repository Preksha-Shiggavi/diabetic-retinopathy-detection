"""
Step 1: Split the APTOS dataset into train/val folders, one sub-folder per class.

Expected input (download from Kaggle, APTOS 2019 Blindness Detection):
    dataset/raw/train.csv
    dataset/raw/train_images/<id_code>.png
"""
import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split
from config import RAW_DIR, TRAIN_DIR, VAL_DIR, FOLDER_NAMES


def copy_images(ids, labels, target_root, image_dir):
    missing = 0
    for i in range(len(ids)):
        src = os.path.join(image_dir, ids[i] + ".png")
        dst_folder = os.path.join(target_root, str(labels[i]))
        os.makedirs(dst_folder, exist_ok=True)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(dst_folder, ids[i] + ".png"))
        else:
            missing += 1
    return missing


def main():
    csv_path = os.path.join(RAW_DIR, "train.csv")
    image_dir = os.path.join(RAW_DIR, "train_images")

    if not os.path.exists(csv_path) or not os.path.isdir(image_dir):
        print("ERROR: put train.csv and the train_images folder inside dataset/raw/")
        return

    df = pd.read_csv(csv_path)
    ids = df["id_code"].tolist()
    labels = df["diagnosis"].tolist()

    tr_ids, va_ids, tr_labels, va_labels = train_test_split(
        ids, labels, test_size=0.2, stratify=labels, random_state=42
    )

    # clean old folders so re-running does not mix data
    for root in (TRAIN_DIR, VAL_DIR):
        if os.path.isdir(root):
            shutil.rmtree(root)

    m1 = copy_images(tr_ids, tr_labels, TRAIN_DIR, image_dir)
    m2 = copy_images(va_ids, va_labels, VAL_DIR, image_dir)

    print("Train images:", len(tr_ids) - m1, "| Val images:", len(va_ids) - m2)
    for i in range(len(FOLDER_NAMES)):
        t = len(os.listdir(os.path.join(TRAIN_DIR, FOLDER_NAMES[i])))
        v = len(os.listdir(os.path.join(VAL_DIR, FOLDER_NAMES[i])))
        print("Class", FOLDER_NAMES[i], "-> train:", t, " val:", v)


if __name__ == "__main__":
    main()
