import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

class AudioProcessing:
    def __init__(self, sample_rate=44100):
        self.sample_rate = sample_rate

    
    def create_spectrogram(self, audio):
        """Version finale avec debug garantie"""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import time
        import os

        # Debug initial
        print("\n=== DÉBUT CRÉATION SPECTROGRAMME ===")
        print(f"Audio reçu: {len(audio)} échantillons")
        print(f"Type: {type(audio)}, Min: {np.min(audio):.2f}, Max: {np.max(audio):.2f}")

        try:
            # 1. Calcul du spectrogramme
            S = librosa.feature.melspectrogram(y=audio, sr=self.sample_rate, n_mels=128, fmax=8000)
            S_dB = librosa.power_to_db(S, ref=np.max)
            print(f"Dimensions spectrogramme: {S_dB.shape}")

            # 2. Création figure
            fig, ax = plt.subplots(figsize=(12, 4))
            img = librosa.display.specshow(S_dB, sr=self.sample_rate, x_axis='time', y_axis='mel', ax=ax)
            plt.colorbar(img, format='%+2.0f dB')
            plt.title('Mel Spectrogram')
            plt.tight_layout()

            # 3. Sauvegarde debug
            debug_dir = "debug_spectrograms"
            os.makedirs(debug_dir, exist_ok=True)
            timestamp = int(time.time())
            debug_path = os.path.join(debug_dir, f"spectro_{timestamp}.png")
            
            # Double sauvegarde (fichier + base64)
            fig.savefig(debug_path, bbox_inches='tight', dpi=100)
            buf = BytesIO()
            fig.savefig(buf, format='png', dpi=100)
            plt.close(fig)

            print(f"=== SPECTROGRAMME CRÉÉ ET SAUVEGARDÉ ===")
            print(f"Chemin: {debug_path}")
            print(f"Taille fichier: {os.path.getsize(debug_path)} octets")

            return base64.b64encode(buf.getvalue()).decode('utf-8')

        except Exception as e:
            print(f"=== ERREUR ===")
            print(str(e))
            # Crée une image d'erreur rouge
            buf = BytesIO()
            Image.new('RGB', (300, 100), color='red').save(buf, format='PNG')
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