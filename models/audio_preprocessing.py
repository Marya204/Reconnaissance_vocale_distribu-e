import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os


class AudioPreprocessing:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate

    def create_spectrogram(self, audio, save_path='static/spectrogramme.png'):
        """Generere mel spectrogram et enregistrer comme fichier PNG avec largeur variable"""
        S = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_mels=128,
            hop_length=128,  
            fmax=8000
        )
        S_dB = librosa.power_to_db(S, ref=np.max)
        
        # Taille variable selon largeur réelle du spectrogramme
        height_px = 128
        width_px = S_dB.shape[1]
        fig_width = max(width_px / 100, 1.28)  # au moins 1.28 pouces de large
        fig_height = height_px / 100  # 1.28 pouces

        plt.figure(figsize=(fig_width, fig_height), dpi=100)
        librosa.display.specshow(
            S_dB,
            sr=self.sample_rate,
            cmap='gray_r'  
        )
        plt.axis('off')  
        plt.tight_layout(pad=0)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()

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

    def audio_to_variable_width_spectrogram(self, filepath, img_height=128):
        y, sr = librosa.load(filepath, sr=self.sample_rate)
        S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=img_height, hop_length=128, fmax=8000)
        S_dB = librosa.power_to_db(S, ref=np.max)

        img = (S_dB - S_dB.min()) / (S_dB.max() - S_dB.min() + 1e-6)
        img = img.astype(np.float32)
        img = img[..., np.newaxis]  # (128, width, 1)
        img = np.expand_dims(img, axis=0)  # (1, 128, width, 1)
        return img


if __name__ == "__main__":
    ap = AudioPreprocessing(sample_rate=16000)
    
    # Nom du fichier audio dans le workspace
    audio_path = "hello_output.wav"  

    if not os.path.exists(audio_path):
        print(f"Fichier non trouvé : {audio_path}")
        exit(1)

    # Charge et génère le spectrogramme PNG
    y, sr = librosa.load(audio_path, sr=16000)
    ap.create_spectrogram(y, save_path='static/spectrogramme.png')
    print(" Spectrogramme sauvegardé dans static/spectrogramme.png")

    # Prépare aussi l'image comme entrée du modèle
    img = ap.audio_to_variable_width_spectrogram(audio_path)
    print("Entrée prête pour le modèle → shape:", img.shape)# (1, 128, variable_width, 1)
  
