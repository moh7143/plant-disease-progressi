from flask import Flask, request, jsonify, send_file
import os
import cv2
import numpy as np
import sqlite3
from datetime import datetime
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import tensorflow as tf

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")
MODEL_DIR = os.path.join(BASE_DIR, "model")
DB_PATH = os.path.join(BASE_DIR, "database.db")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ----------- Model paths -----------------
CROP_MODELS_PATHS = {
    "apple": os.path.join(MODEL_DIR, "apple_.h5"),
    "corn": os.path.join(MODEL_DIR, "corn_.h5"),
    "strawberry": os.path.join(MODEL_DIR, "strawberry_.h5"),
    "tomato": os.path.join(MODEL_DIR, "tomato_.h5"),
    "potato": os.path.join(MODEL_DIR, "potato_.h5"),
    "grapes": os.path.join(MODEL_DIR, "grapes_.h5"),
    "peach": os.path.join(MODEL_DIR, "peach_.h5"),
    "cherry": os.path.join(MODEL_DIR, "cherry_.h5")
}

loaded_models = {}
CLASSES = ["Apple Scab", "Black Rot", "Cedar Rust", "Healthy"]

# ------------- SQLite Database -----------------
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS predictions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        crop TEXT,
        disease TEXT,
        stage TEXT,
        severity REAL,
        time TEXT
    )
    """)
    conn.commit()
    conn.close()

init_db()

# ------------- Severity Calculation --------------
def calculate_severity(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

    severity = (np.mean(gray) + np.mean(hsv[:, :, 1]) + np.mean(lab[:, :, 0])) / 3
    severity = min(100, max(0, severity / 2))

    if severity < 20: stage = "Healthy / Mild"
    elif severity < 40: stage = "Moderate"
    elif severity < 70: stage = "Severe"
    else: stage = "Critical"

    return round(severity, 2), stage

# ------------- Treatment Recommendations -------
def get_treatment(disease, stage):
    data = {
        "Apple Scab": {
            "Healthy / Mild": "Regular monitoring and watering.",
            "Moderate": "Apply fungicide and remove infected leaves.",
            "Severe": "Systemic fungicide + pruning.",
            "Critical": "Remove plant to prevent spread."
        },
        "Healthy": {"Healthy / Mild": "Plant is healthy."}
    }
    return data.get(disease, {}).get(stage, "Monitor plant health and ensure proper nutrition.")

# ------------- Grad-CAM -------------------------
def generate_gradcam(model, img_array, layer_name=None):
    if layer_name is None:
        layer_name = model.layers[-3].name

    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        class_idx = tf.argmax(predictions[0])
        loss = predictions[:, class_idx]

    grads = tape.gradient(loss, conv_outputs)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

# ------------- /predict route -------------------
@app.route("/predict", methods=["POST"])
def predict():
    crop_type = request.form.get("crop", "apple").lower()
    if crop_type not in CROP_MODELS_PATHS:
        return jsonify({"error": f"Model for {crop_type} not found"}), 400

    if crop_type not in loaded_models:
        try: loaded_models[crop_type] = load_model(CROP_MODELS_PATHS[crop_type])
        except Exception as e: return jsonify({"error": str(e)}), 500

    file = request.files["file"]
    filename = file.filename
    file_path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(file_path)

    img = cv2.imread(file_path)
    img_resized = cv2.resize(img, (224, 224)) / 255.0
    img_array = np.expand_dims(img_resized, axis=0)

    pred = loaded_models[crop_type].predict(img_array)
    disease = CLASSES[np.argmax(pred)]

    severity, stage = calculate_severity(file_path)
    treatment = get_treatment(disease, stage)

    # ---------- Grad-CAM ----------
    heatmap = generate_gradcam(loaded_models[crop_type], img_array)
    heatmap = cv2.resize(heatmap, (224, 224))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    original = cv2.resize(img, (224, 224))
    superimposed = cv2.addWeighted(original, 0.6, heatmap, 0.4, 0)
    gradcam_path = os.path.join(UPLOAD_FOLDER, f"gradcam_{filename}")
    cv2.imwrite(gradcam_path, superimposed)

    # -------- Store in SQLite (optional) -------
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("INSERT INTO predictions (crop,disease,stage,severity,time) VALUES (?,?,?,?,?)",
                   (crop_type, disease, stage, severity, datetime.now().isoformat()))
    conn.commit()
    conn.close()

    return jsonify({
        "crop": crop_type,
        "disease": disease,
        "stage": stage,
        "severity_percent": severity,
        "treatment": treatment,
        "gradcam_url": f"/gradcam/{os.path.basename(gradcam_path)}"
    })

# ------------- Grad-CAM Image Route -------------
@app.route("/gradcam/<filename>")
def gradcam_image(filename):
    path = os.path.join(UPLOAD_FOLDER, filename)
    return send_file(path, mimetype="image/png")

# ------------- /history ------------------------
@app.route("/history")
def history():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT crop, disease, stage, severity, time FROM predictions ORDER BY time DESC")
    rows = cursor.fetchall()
    conn.close()
    return jsonify([{"crop": r[0], "disease": r[1], "stage": r[2], "severity": r[3], "time": r[4]} for r in rows])

# ------------- /chart --------------------------
@app.route("/chart")
def chart():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT severity FROM predictions")
    data = cursor.fetchall()
    conn.close()
    if not data: return jsonify({"error": "No data"})
    plt.figure()
    plt.plot([d[0] for d in data])
    plt.title("Disease Severity Progression")
    chart_path = os.path.join(UPLOAD_FOLDER, "chart.png")
    plt.savefig(chart_path)
    plt.close()
    return send_file(chart_path, mimetype="image/png")

if __name__ == "__main__":
    app.run(debug=True)
