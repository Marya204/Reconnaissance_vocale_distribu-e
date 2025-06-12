import unittest
import numpy as np
from models.audio_preprocessing import AudioProcessing  # Vérifie le chemin si besoin

class TestAudioProcessing(unittest.TestCase):
    def setUp(self):
        self.processor = AudioProcessing()
        self.test_audio = np.random.rand(44100)  # 1 seconde d'audio aléatoire à 44.1kHz

    def test_normalize_audio(self):
        normalized = self.processor.normalize(self.test_audio)
        self.assertTrue(np.max(np.abs(normalized)) <= 1.0)

    def test_create_spectrogram(self):
        spectrogram = self.processor.create_spectrogram(self.test_audio)
        self.assertIsInstance(spectrogram, str)
        self.assertGreater(len(spectrogram), 1000)  # On attend une image encodée base64 non vide

    def test_extract_mfcc(self):
        mfcc = self.processor.extract_mfcc(self.test_audio)
        self.assertIsInstance(mfcc, list)
        self.assertEqual(len(mfcc), 20)  # On a demandé 20 coefficients MFCC

if __name__ == '__main__':
    unittest.main()
