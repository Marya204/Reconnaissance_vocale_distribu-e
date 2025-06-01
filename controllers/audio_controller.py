from models.audio_capture import AudioCapture  
from models.audio_preprocessing import AudioProcessing 
import io
import soundfile as sf
import base64

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
        # Save to buffer
        buffer = io.BytesIO()
        sf.write(buffer, audio, self.capture.sample_rate, format='WAV')
        buffer.seek(0)
        audio_bytes = buffer.read()
        audio_base64 = base64.b64encode(audio_bytes).decode('utf-8')
        audio_data_url = f"data:audio/wav;base64,{audio_base64}"

        return {
            'audio': [float(x) for x in audio],
            'audio_data_url': audio_data_url,
            'spectrogram': self.processor.create_spectrogram(audio),
            'mfcc': self.processor.extract_mfcc(audio),
            'sample_rate': self.capture.sample_rate
        }