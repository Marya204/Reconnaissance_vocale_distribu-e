import unittest
import numpy as np
from models.audio_preprocessing import AudioProcessing # type: ignore

class TestAudioProcessing(unittest.TestCase):
    def setUp(self):
        self.processor = AudioProcessing()
        self.test_audio = np.random.rand(44100)  # 1 second of random audio at 44.1kHz
    
    def test_normalize_audio(self):
        normalized = self.processor.normalize_audio(self.test_audio)
        self.assertTrue(np.max(np.abs(normalized)) <= 1.0)
        
    def test_remove_noise(self):
        cleaned = self.processor.remove_noise(self.test_audio)
        self.assertEqual(len(cleaned), len(self.test_audio))
        
    def test_create_spectrogram(self):
        spectrogram = self.processor.create_spectrogram(self.test_audio)
        self.assertIsInstance(spectrogram, str)
        self.assertTrue(spectrogram.startswith('iVBOR'))  # PNG base64 prefix
        
    def test_extract_mfcc(self):
        mfcc = self.processor.extract_mfcc(self.test_audio)
        self.assertEqual(mfcc.shape[0], 13)  # Default n_mfcc

if __name__ == '__main__':
    unittest.main()