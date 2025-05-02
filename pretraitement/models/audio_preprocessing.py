import librosa
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class AudioProcessing:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate
    
    def create_spectrogram(self, audio):
        """Generate mel spectrogram"""
        S = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=128,
            fmax=8000
        )
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        plt.figure(figsize=(12, 4))
        librosa.display.specshow(
            S_dB,
            sr=self.sample_rate,
            x_axis='time',
            y_axis='mel'
        )
        plt.colorbar(format='%+2.0f dB')
        plt.title('Mel Spectrogram (30s)')
        plt.tight_layout()
        
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close()
        return base64.b64encode(buf.getvalue()).decode('utf-8')
    
    def extract_mfcc(self, audio):
        """Extract 20 MFCC features"""
        return librosa.feature.mfcc(
            y=audio,
            sr=self.sample_rate,
            n_mfcc=20
        ).tolist()
    
    def normalize(self, audio):
        """Normalize audio"""
        return librosa.util.normalize(audio)