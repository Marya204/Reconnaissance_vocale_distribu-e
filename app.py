from fastapi import FastAPI, Request, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import os
from werkzeug.utils import secure_filename
from controllers.audio_controller import AudioController  # type: ignore
from views.audio_view import AudioView  # type: ignore
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from starlette.middleware.cors import CORSMiddleware
from models.inference import ASRInference
import glob


os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_FOLDER = 'tmp_uploads'
MAX_CONTENT_LENGTH = 30 * 1024 * 1024  # 30MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

controller = AudioController()
view = AudioView()

# --- Partie existante ---

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return view.home(request)

@app.post("/record")
async def record_audio():
    try:
        data = controller.record_and_process()
        return view.audio_data(data)
    except Exception as e:
        return view.error(str(e))

@app.post("/process")
async def process_audio(background_tasks: BackgroundTasks, audio: UploadFile = File(...)):
    if not audio:
        return view.error("No audio file uploaded")

    if audio.filename == '':
        return view.error("No file selected")

    try:
        filename = secure_filename(audio.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)

        with open(filepath, "wb") as f:
            f.write(await audio.read())

        data = controller.process_file(filepath)

        if 'spectrogram' in data and not isinstance(data['spectrogram'], str):
            import base64
            from io import BytesIO
            from PIL import Image

            if isinstance(data['spectrogram'], np.ndarray):
                spectrogram = (data['spectrogram'] * 255).astype(np.uint8)
                img = Image.fromarray(spectrogram)
            else:
                img = Image.fromarray(data['spectrogram'])

            buffered = BytesIO()
            img.save(buffered, format="PNG")
            data['spectrogram'] = base64.b64encode(buffered.getvalue()).decode('utf-8')

        return view.audio_data(data)

    except Exception as e:
        return view.error(str(e))

    finally:
        background_tasks.add_task(os.remove, filepath)



# Chargement du modèle ASR
ASR_MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "saved_model", "saved_model.keras")

asr_model = None
try:
    if os.path.exists(ASR_MODEL_PATH):
        asr_model = tf.keras.models.load_model(ASR_MODEL_PATH)
        print("✅ Modèle ASR chargé avec succès")
    else:
        print(f"❌ Fichier du modèle introuvable: {ASR_MODEL_PATH}")
except Exception as e:
    print(f"❌ Erreur lors du chargement du modèle ASR: {str(e)}")
inference = ASRInference(model_path=ASR_MODEL_PATH)
# Dictionnaire et fonction de décodage
int_to_char = {i + 1: c for i, c in enumerate("abcdefghijklmnopqrstuvwxyz ")}

def decode_prediction(pred):
    if pred is None or len(pred) == 0:
        return ""
    pred_indices = np.argmax(pred, axis=-1)
    decoded = ""
    prev = -1
    for idx in pred_indices:
        if idx != prev and idx != 0:
            decoded += int_to_char.get(idx, "")
        prev = idx
    return decoded

@app.post("/transcribe")
async def transcribe(background_tasks: BackgroundTasks, spectrogram: UploadFile = File(...)):
    if not spectrogram:
        return JSONResponse(content={"error": "No spectrogram file uploaded"}, status_code=400)

    if spectrogram.filename == '':
        return JSONResponse(content={"error": "No file selected"}, status_code=400)

    filename = secure_filename(spectrogram.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)

    with open(filepath, "wb") as f:
        f.write(await spectrogram.read())

    try:
        img = load_img(filepath, color_mode="grayscale", target_size=(128, 128))
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        prediction = asr_model(img_array)
        predicted_text = decode_prediction(prediction.numpy()[0])

        return JSONResponse(content={"transcription": predicted_text})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
    finally:
        background_tasks.add_task(os.remove, filepath)
@app.get("/transcribe/latest")
async def transcribe_latest():
    print("Appel à /transcribe/latest")
    folder_path = "debug_spectrograms"  # Ton dossier de spectrogrammes
    files = glob.glob(os.path.join(folder_path, "*.png"))
    print(f"Recherche fichiers dans {folder_path} : {files}")
    if not files:
        print("Aucun fichier dans debug_spectrograms")
        return JSONResponse(content={"error": "Aucun fichier trouvé"}, status_code=404)

    latest_file = max(files, key=os.path.getmtime)
    print(f"Fichier le plus récent : {latest_file}")
    try:
        predicted_text, _ = inference.predict(latest_file)
        print(f"Transcription : {predicted_text}")
        return {
            "filename": os.path.basename(latest_file),
            "transcription": predicted_text
        }
    except Exception as e:
        print(f"Erreur prediction: {str(e)}")
        return JSONResponse(content={"error": str(e)}, status_code=500)

# --- Démarrage automatique dans un navigateur (à faire via uvicorn) ---
if __name__ == '__main__':
    port = 5000
    url = f"http://localhost:{port}"

    def open_browser():
        import webbrowser
        import time
        time.sleep(1)
        webbrowser.open_new(url)

    import threading
    threading.Thread(target=open_browser).start()

    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)
