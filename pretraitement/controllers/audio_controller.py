from models.audio_capture import AudioCapture # type: ignore
from models.audio_preprocessing import AudioProcessing # type: ignore

class AudioController:
    def __init__(self):
        self.capture = AudioCapture(duration=30)
        self.processor = AudioProcessing()
    
    def record_and_process(self):
        audio = self.capture.record_audio()
        return self._process_audio(audio)
    
    def process_file(self, filepath):
        audio = self.capture.load_audio(filepath)
        return self._process_audio(audio)
    
    def _process_audio(self, audio):
        audio = self.processor.normalize(audio)
        return {
            'audio': audio.tolist(),
            'spectrogram': self.processor.create_spectrogram(audio),
            'mfcc': self.processor.extract_mfcc(audio),
            'sample_rate': self.capture.sample_rate
        }