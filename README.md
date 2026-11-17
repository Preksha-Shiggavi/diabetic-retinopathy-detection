# AI-Based Diabetic Retinopathy Detection

Classifies a retina (fundus) image into 5 grades:
0 No DR, 1 Mild, 2 Moderate, 3 Severe, 4 Proliferative DR.

## Setup (once)
1. Install Python 3.10, 3.11 or 3.12 (tick "Add Python to PATH").
2. Open this folder in VS Code terminal and run:
   python -m venv venv
   venv\Scripts\activate          (Windows)   |   source venv/bin/activate  (Mac/Linux)
   pip install -r requirements.txt

## Dataset
Download "APTOS 2019 Blindness Detection" from Kaggle (free account).
Put these into dataset/raw/ :
   train.csv
   train_images/   (folder with all .png files)

## Run in this order
   python prepare_data.py     # splits images into train/val class folders
   python train.py            # trains and saves models/dr_model.keras
   python evaluate.py         # accuracy report + confusion matrix
   python app.py              # open http://127.0.0.1:5000
   python predict.py some_eye.png   # optional: test from terminal

## Tips
- No GPU? Training works on CPU but is slow. Use Google Colab (free GPU) for train.py,
  then download models/dr_model.keras into this folder's models/ directory.
- Slow laptop? Lower EPOCHS_HEAD / EPOCHS_FINE in config.py or set IMG_SIZE to 160.
