import librosa
import noisereduce as nr
import numpy as np
import soundfile as sf
import requests
import io
import base64

class InferlessPythonModel:
    def initialize(self):
        self.threshold = 0.01

    def download_audio(self, url):
        response = requests.get(url)
        audio_data = io.BytesIO(response.content)
        return audio_data

    def load_audio(self, file_content):
        audio, sr = librosa.load(file_content, sr=None)
        return audio, sr

    def analyze_and_reduce_noise(self, audio, sr):
        energy = np.sqrt(np.mean(audio**2))
        if energy > self.threshold:
            print("Applying noise reduction...")
            audio_clean = nr.reduce_noise(audio, sr=sr)
        else:
            print("Audio is clean. Skipping noise reduction.")
            audio_clean = audio
        return audio_clean

    def encode_audio_base64(self, audio, sr):
        with io.BytesIO() as audio_buffer:
            sf.write(audio_buffer, audio, sr, format='wav')
            audio_base64 = base64.b64encode(audio_buffer.getvalue()).decode('utf-8')
        return audio_base64

    def infer(self, inputs):
        file_name = inputs["file_name"]
        file_content = self.download_audio(file_name)
        
        audio, sr = self.load_audio(file_content)
        audio_clean = self.analyze_and_reduce_noise(audio, sr)
        audio_base64 = self.encode_audio_base64(audio_clean, sr)
        
        return {"audio_base64": audio_base64}

    def finalize(self):
        pass
