import os
import time
import gdown
import numpy as np
from flask import Flask, request, render_template
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)
app.config['TEMPLATES_AUTO_RELOAD'] = True

# ===============================
# DOWNLOAD MODEL (optional)
# ===============================
if not os.path.exists("model.h5"):
    url = "https://drive.google.com/uc?id=1uQ3FTjqmEIyk5WDHeS0FVItIngIZHKlP"
    gdown.download("url", "model.h5", quiet=False)

# ===============================
# LOAD MODEL
# ===============================
model = load_model("model.h5")
print("✅ Model loaded successfully!")

# ===============================
# CLASS LABELS
# ===============================
classes = ['Healthy', 'Leaf_spot', 'Powdery', 'Rust']

# ===============================
# UPLOAD FOLDER
# ===============================
UPLOAD_FOLDER = "static/uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# ===============================
# HOME PAGE
# ===============================
@app.route('/')
def home():
    return render_template('index.html')

# ===============================
# PREDICTION
# ===============================
@app.route('/predict', methods=['POST'])
def predict():

    if 'file' not in request.files:
        return "No file uploaded"

    file = request.files['file']

    if file.filename == '':
        return "No file selected"

    # Unique filename
    filename = str(int(time.time())) + "_" + secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Preprocess image
    img = image.load_img(filepath, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0) / 255.0

    # Prediction
    preds = model.predict(img)[0]

    class_index = np.argmax(preds)
    final_class = classes[class_index]
    confidence = round(float(preds[class_index]) * 100, 2)

    probabilities = {
        classes[i]: round(float(preds[i]) * 100, 2)
        for i in range(len(classes))
    }

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