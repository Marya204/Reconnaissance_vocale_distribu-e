import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

IMG_HEIGHT = 128
ALPHABET = "abcdefghijklmnopqrstuvwxyz "
BLANK_LABEL = len(ALPHABET)  # index du 'blank' pour CTC

class ASRInference:
    def __init__(self, model_path):
        # Charger le modèle complet (avec Lambda CTC)
        full_model = load_model(
            model_path,
            compile=False
            # safe_mode=False  # décommenter si erreur de chargement
        )
        print("Modèle chargé")

        # Tenter d'extraire la couche nommée "time_distributed"
        try:
            self.base_model = tf.keras.Model(
                inputs=full_model.inputs,
                outputs=full_model.get_layer('time_distributed').output
            )
            print("Couche 'time_distributed' utilisée comme sortie.")
        except ValueError:
            print("Couche 'time_distributed' non trouvée, fallback vers la dernière couche.")
            self.base_model = tf.keras.Model(
                inputs=full_model.inputs,
                outputs=full_model.layers[-1].output
            )

    def preprocess_image(self, image_path):
        img = Image.open(image_path).convert('L')  # grayscale
        img = np.array(img, dtype=np.float32) / 255.0
        print(f"Spectrogram shape before expand_dims: {img.shape}")
        img = np.expand_dims(img, axis=-1)  # (H, W, 1)
        img = np.expand_dims(img, axis=0)   # (1, H, W, 1)
        print(f"Spectrogram shape after expand_dims: {img.shape}")
        return img

    def decode_predictions(self, preds):
        input_len = np.ones(preds.shape[0]) * preds.shape[1]
        results = tf.keras.backend.ctc_decode(preds, input_length=input_len, greedy=True)[0][0]
        results = tf.keras.backend.get_value(results)

        decoded_texts = []
        for res in results:
            text = ''.join([ALPHABET[c] for c in res if c != -1 and c != BLANK_LABEL])
            if text == "":
                text = "[aucune transcription reconnue]"
            decoded_texts.append(text)
        return decoded_texts

    def infer(self, image_path):
        processed_image = self.preprocess_image(image_path)
        preds = self.base_model.predict(processed_image)
        print("Preds shape:", preds.shape)
        print("Preds sample:", preds[0, :10, :5])
        print("Preds max:", np.max(preds), "Preds min:", np.min(preds))

        decoded_text = self.decode_predictions(preds)
        print("Texte transcrit :", decoded_text[0])
        return decoded_text[0]
