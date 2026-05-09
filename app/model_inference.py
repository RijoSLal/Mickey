import os 
os.environ["CUDA_VISIBLE_DEVICES"]="0" #if you want to run this in GPU change -1 to 1 or 2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import librosa
import numpy as np
from tensorflow import keras 
from keras.models import load_model
import joblib
import sys
from pathlib import Path

# Add the app directory to sys.path
app_dir = Path(__file__).resolve().parent
if str(app_dir) not in sys.path:
    sys.path.append(str(app_dir))

from extraction import extract_features
import logging


from typing import Union, Optional
import io

class Verify:
    def __init__(self, audio_source: Union[str, io.BytesIO]):
        self.audio_source = audio_source

    def classify_music(self) -> bool:
        try:
            if isinstance(self.audio_source, io.BytesIO):
                self.audio_source.seek(0)
            audio, sr = librosa.load(self.audio_source, sr=None)

            if self.detect_silence(audio, sr):
                 return False  # no music detected


            zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
            energy = np.mean(librosa.feature.rms(y=audio))


            # ZCR typically higher for music, energy significant for music

            return True if zcr > 0.05 and energy > 0.02 else False

        except Exception as e:
            logging.error(f"error processing audio file: {e}")
            return False  #  no music if processing fails

    def detect_silence(self, audio: np.ndarray, sr: int, threshold: float = 0.01, duration: float = 1.0) -> bool:

        """
        detect if the majority of the audio is silence.

        returns:
        - true if silence detected else false

        """
        frame_length = int(sr * duration)
        hop_length = frame_length // 2

        # calculate energy across frames
        energy = librosa.feature.rms(y=audio, frame_length=frame_length, hop_length=hop_length)[0]

        # check the proportion of silent frames
        silent_frames = np.sum(energy < threshold)
        total_frames = len(energy)

        # consider audio silent if more than 70% of frames are below the threshold
        return silent_frames / total_frames > 0.7



from pathlib import Path

# Load model and scaler once
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = str(BASE_DIR / 'models/music_model.h5')
SCALER_PATH = str(BASE_DIR / 'models/scaler.pkl')

if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
    model = load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
else:
    model = None
    scaler = None
    logging.error(f"Model or scaler file not found at {MODEL_PATH} or {SCALER_PATH}!")

def model_prediction(audio_source: Union[str, io.BytesIO]) -> str:

    """
     uses class check methods to check whether the audio is actually music or not and predict the emotion

    """

    music_check = Verify(audio_source)

    if music_check.classify_music() == True:
        if model is None or scaler is None:
            return "Model or scaler not loaded"

        features = extract_features(audio_source)

        music = scaler.transform(features.reshape(1, -1))
        final = music.reshape(1, 1, 57)
        predicted = model.predict(final)

        val = np.argmax(predicted)
        emotion_array = ["Sad", "Serenity", "Nostalgia", "Joy", "Confidence", "Relax", "Angry", "Happy", "Chill", "Excitement"]
        return emotion_array[val]
    else:
        return "No audio detected"



