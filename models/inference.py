import tensorflow as tf
import numpy as np
from PIL import Image

# === Dictionnaire d'encodage ===
CHARS = "abcdefghijklmnopqrstuvwxyz '"
INDEX_TO_CHAR = {i + 1: c for i, c in enumerate(CHARS)}  # 0 = CTC blank


class ASRInference:
    def __init__(self, model_path="saved_model/asr_model"):
        self.model = tf.keras.models.load_model(model_path)

    def _decode_prediction(self, pred):
        input_len = np.ones(pred.shape[0]) * pred.shape[1]
        results = tf.keras.backend.ctc_decode(pred, input_length=input_len, greedy=True)[0][0]
        output_texts = []
        for res in results.numpy():
            text = ''.join([INDEX_TO_CHAR.get(r, '') for r in res])
            output_texts.append(text)
        return output_texts

    def predict(self, image_path):
        img = Image.open(image_path).convert('L').resize((128, 128))
        img = np.array(img) / 255.0
        img = img.reshape(1, 128, 128, 1).astype(np.float32)

        preds = self.model.predict(img)
        texts = self._decode_prediction(preds)
        return texts[0], None  # Ajoute confidence si tu la calcules plus tard
