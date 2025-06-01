from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
from controllers.audio_controller import AudioController  # type: ignore
from views.audio_view import AudioView  # type: ignore
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'tmp_uploads'
app.config['MAX_CONTENT_LENGTH'] = 30 * 1024 * 1024  # 30MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

controller = AudioController()
view = AudioView()

# --- Partie existante ---

@app.route('/')
def home():
    return view.home()

@app.route('/record', methods=['POST'])
def record_audio():
    try:
        data = controller.record_and_process()
        return view.audio_data(data)
    except Exception as e:
        return view.error(str(e))

@app.route('/process', methods=['POST'])
def process_audio():
    if 'audio' not in request.files:
        return view.error("No audio file uploaded")

    file = request.files['audio']
    if file.filename == '':
        return view.error("No file selected")

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            data = controller.process_file(filepath)
            return view.audio_data(data)
        except Exception as e:
            return view.error(str(e))
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)

# --- Nouvelle route pour ta partie modèle ASR ---

# Chargement du modèle ASR (une seule fois au lancement)
asr_model = tf.saved_model.load("saved_model/")

# Dictionnaire inverse pour décoder la sortie
int_to_char = {i + 1: c for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ")}

def decode_prediction(pred):
    pred_indices = np.argmax(pred, axis=-1)
    decoded = ""
    prev = -1
    for idx in pred_indices:
        if idx != prev and idx != 0:
            decoded += int_to_char.get(idx, "")
        prev = idx
    return decoded

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'spectrogram' not in request.files:
        return jsonify({"error": "No spectrogram file uploaded"}), 400

    file = request.files['spectrogram']
    if file.filename == '':
        return jsonify({"error": "No file selected"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        img = load_img(filepath, color_mode="grayscale", target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = asr_model(img_array)
        predicted_text = decode_prediction(prediction.numpy()[0])

        return jsonify({"transcription": predicted_text})
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)

if __name__ == '__main__':
    app.run(debug=True)
