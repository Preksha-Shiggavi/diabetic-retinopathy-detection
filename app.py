"""
Step 4: Web app. Run:  python app.py   then open http://127.0.0.1:5000
"""
import os
import uuid
from flask import Flask, render_template, request, jsonify
from config import UPLOAD_DIR, MODEL_PATH
from predict import predict_image

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB limit
ALLOWED = ("png", "jpg", "jpeg")
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if not os.path.exists(MODEL_PATH):
        return jsonify({"error": "Model not trained yet. Run train.py first."}), 400
    file = request.files.get("image")
    if file is None or file.filename == "":
        return jsonify({"error": "No image selected."}), 400
    ext = file.filename.rsplit(".", 1)[-1].lower()
    if ext not in ALLOWED:
        return jsonify({"error": "Only PNG/JPG images allowed."}), 400

    path = os.path.join(UPLOAD_DIR, str(uuid.uuid4()) + "." + ext)
    file.save(path)
    try:
        return jsonify(predict_image(path))
    finally:
        os.remove(path)


if __name__ == "__main__":
    app.run(debug=True)
