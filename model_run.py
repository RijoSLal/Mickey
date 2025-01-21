import os 
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #if you want to run this in GPU change -1 to 1 or 2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import librosa
import numpy as np
from tensorflow import keras 
from keras.models import load_model
import joblib
from extraction import extract_features




class Check:
    def __init__(self, audio_path):
        self.audio_path = audio_path

    def classify_music(self):
        try:
            audio, sr = librosa.load(self.audio_path, sr=None)
            
          
            if self.detect_silence(audio, sr):
                return False  # no music detected

         
            zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio))
            energy = np.mean(librosa.feature.rms(y=audio))
            
          
            # ZCR typically higher for music, energy significant for music

            if zcr > 0.05 and energy > 0.02:
                return True  # music detected
            else:
                return False  # no music detected

        except Exception as e:
            print(f"Error processing the audio file: {e}")
            return False  #  no music if processing fails

    def detect_silence(self, audio, sr, threshold=0.01, duration=1.0):

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



def model_prediction_with_temp(path_for_real_audio):

    """
     uses class check methods to check whether the audio is actually music or not and predict the emotion
     
    """
    
    music_check=Check(path_for_real_audio)
    
    if music_check.classify_music()==True:

        model = load_model('model_scaler/music_model.h5')
        scaler = joblib.load('model_scaler/scaler.pkl')

        features=extract_features(path_for_real_audio)
        music=scaler.fit_transform(features.reshape(-1,1))
        final=music.reshape(1, 1, 57)
        predicted=model.predict(final)
        val=np.argmax(predicted)
        emotion_array=['Sad', 'Serenity', 'Nostalgia', 'Joy', 'Confidence' ,"Relax","Angry","Happy",'Chill',"Excitement"]
        return emotion_array[val]
    else:
        return "No audio detected"



