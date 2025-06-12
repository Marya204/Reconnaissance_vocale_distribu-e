# components/inference_component.py

import tensorflow as tf
import os

def inference_step(audio_path: str, model_dir: str, output_text: str):
    import numpy as np
    import librosa

    # Load audio
    audio, sr = librosa.load(audio_path, sr=None)
    # (Supposons ici que ton modèle prend un spectrogramme ou MFCC — adapte si besoin)

    # Charger le modèle
    model = tf.keras.models.load_model(model_dir)

    # Prétraitement minimal (à adapter selon ton modèle)
    mfcc = librosa.feature.mfcc(audio, sr=sr)
    mfcc = np.expand_dims(mfcc, axis=0)

    # Prédiction
    prediction = model.predict(mfcc)

    # Décodage (à adapter selon la sortie du modèle)
    predicted_text = "..."  # TODO: mettre la logique de décodage

    with open(output_text, "w") as f:
        f.write(predicted_text)
