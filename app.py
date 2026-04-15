import os
import uuid
import numpy as np
import tensorflow as tf

from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from tensorflow.keras.preprocessing import image

# ===============================
# LOAD TFLITE MODEL
# ===============================
MODEL_PATH = "model.tflite"

if not os.path.exists(MODEL_PATH):
    print("❌ model.tflite not found!")
    exit()

print("🔄 Loading TFLite model...")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("✅ TFLite model loaded!")

# ===============================
# FLASK APP
# ===============================
app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# Disable cache
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-store'
    return response

# ===============================
# CLASSES
# ===============================
classes = ['Healthy', 'Leaf_spot', 'Powdery', 'Rust']

# ===============================
# UPLOAD FOLDER
# ===============================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ===============================
# HOME
# ===============================
@app.route('/')
def home():
    return render_template('index.html')

# ===============================
# PREDICT
# ===============================
@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    if file.filename == '':
        return "No file selected"

    # Save file
    filename = str(uuid.uuid4()) + "_" + secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # ===============================
    # IMAGE PROCESSING (PIL - BEST)
    # ===============================
    img = image.load_img(filepath, target_size=(225, 225))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0

    # Match dtype
    img = img.astype(input_details[0]['dtype'])

    # ===============================
    # TFLITE PREDICTION
    # ===============================
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    preds = interpreter.get_tensor(output_details[0]['index'])[0]

    # ===============================
    # NORMALIZE OUTPUT
    # ===============================
    preds = preds.astype(float)

    # Avoid zero dominance
    preds = preds + 1e-6  

    total = np.sum(preds)
    preds = preds / total

    probabilities = {
        classes[i]: round(preds[i] * 100, 2)
        for i in range(len(classes))
    }

    # ===============================
    # FINAL CLASS LOGIC
    # ===============================
    final_class = max(probabilities, key=probabilities.get)
    confidence = probabilities[final_class]

    # If low confidence → uncertain
    if confidence < 50:
        final_class = "Mixed / Uncertain"

    img_path = 'uploads/' + filename

    return render_template(
        'result.html',
        prediction=final_class,
        confidence=confidence,
        probabilities=probabilities,
        img_path=img_path
    )

# ===============================
# RUN APP
# ===============================
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 10000))
    app.run(host="0.0.0.0", port=port, debug=True)