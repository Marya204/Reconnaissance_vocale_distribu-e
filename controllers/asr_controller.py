
from models.inference import ASRInference

class ASRController:
    def __init__(self):
        self.inference = ASRInference()

    def predict(self, path):
        text, confidence = self.inference.predict(path)
        return text, confidence

