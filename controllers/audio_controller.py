# controller/audio_controller.py
import librosa
from models.audio_preprocessing import AudioProcessing
from werkzeug.utils import secure_filename
import os

class AudioController:
    def __init__(self):
        self.processor = AudioProcessing()
    
    def process_file(self, filepath):
        # Chargement de l'audio
        audio, sr = librosa.load(filepath, sr=None)
        
        # Normaliser l'audio
        normalized_audio = self.processor.normalize(audio)
        
        # Générer spectrogramme
        spectrogram_base64 = self.processor.create_spectrogram(normalized_audio)
        
        # Extraire les MFCC si nécessaire
        mfcc = self.processor.extract_mfcc(normalized_audio)
        
        # Renvoie les résultats du traitement
        return {
            'spectrogram': spectrogram_base64,
            'mfcc': mfcc
        }
