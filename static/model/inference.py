import tensorflow as tf
import numpy as np
from PIL import Image

class ASRInference:
    def __init__(self):
        self.model = tf.keras.models.load_model("saved_model")

    def predict(self, path):
        img = Image.open(path).convert('L').resize((128,128))
        x = np.expand_dims(np.array(img)/255.0, axis=(0,-1))
        y_pred = self.model.predict(x)[0]
        tokens = np.argmax(y_pred, axis=-1)
        chars = "abcdefghijklmnopqrstuvwxyz '"
        text = ''.join([chars[t] for t in tokens if t < len(chars)])
        confidence = float(np.mean(np.max(y_pred, axis=-1)))
        return text, confidence
