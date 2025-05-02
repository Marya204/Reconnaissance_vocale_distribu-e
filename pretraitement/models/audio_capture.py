import sounddevice as sd
import numpy as np
import librosa
from scipy.io.wavfile import write

class AudioCapture:
    def __init__(self, sample_rate=44100, duration=30):
        self.sample_rate = sample_rate
        self.duration = duration
        self.audio_data = None
        
    def record_audio(self):
        """Record audio for 30 seconds"""
        print(f"Recording for {self.duration} seconds...")
        self.audio_data = sd.rec(
            int(self.duration * self.sample_rate),
            samplerate=self.sample_rate,
            channels=1,
            dtype='float32'
        )
        sd.wait()
        return self.audio_data.flatten()
    
    def load_audio(self, file_path):
        """Load audio with exactly 30s duration"""
        audio, sr = librosa.load(file_path, sr=self.sample_rate, mono=True)
        
        # Ensure 30s duration
        if len(audio) > self.sample_rate * self.duration:
            audio = audio[:self.sample_rate * self.duration]
        elif len(audio) < self.sample_rate * self.duration:
            audio = np.pad(audio, (0, self.sample_rate * self.duration - len(audio)))
            
        self.audio_data = audio
        return audio