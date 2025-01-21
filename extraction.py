import os
import librosa
import pandas as pd
import numpy as np




def extract_features(file_path):
    
    """

    this function extracts audio features from a given audio file 
    and uses mirror padding to handle short audio signals


    """

    y, sr = librosa.load(file_path, sr=None)

   
    target_length = 30 * sr  
    if len(y) < target_length:
        padding = target_length - len(y)
        y = np.pad(y, (0, padding), mode='reflect')
    else:
        y = y[:target_length]

    chroma_stft = librosa.feature.chroma_stft(y=y, sr=sr)
    rms = librosa.feature.rms(y=y)
    spectral_centroid = librosa.feature.spectral_centroid(y=y, sr=sr)
    spectral_bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zero_crossing_rate = librosa.feature.zero_crossing_rate(y=y)
    harmony, perceptr = librosa.effects.harmonic(y), librosa.effects.percussive(y)
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

   
    features = [
        float(np.mean(chroma_stft)), float(np.var(chroma_stft)),
        float(np.mean(rms)), float(np.var(rms)),
        float(np.mean(spectral_centroid)), float(np.var(spectral_centroid)),
        float(np.mean(spectral_bandwidth)), float(np.var(spectral_bandwidth)),
        float(np.mean(rolloff)), float(np.var(rolloff)),
        float(np.mean(zero_crossing_rate)), float(np.var(zero_crossing_rate)),
        float(np.mean(harmony)), float(np.var(harmony)),
        float(np.mean(perceptr)), float(np.var(perceptr)),
        float(tempo)
    ]

  
    for i in range(20):
        features.append(float(np.mean(mfcc[i])))
        features.append(float(np.var(mfcc[i])))

    return np.array(features, dtype=float)





def process_audio_folder(folder_path, label=None):
    
    """

    this function go through all files in the given folder and if the file format is mp3 or wav it passes that file through 
    extract_features function and returns labels and the preprocessed data
    
    
    """

    data = []
    labels = []
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".mp3") or filename.endswith(".wav"):
         
            audio_path = os.path.join(folder_path, filename)
            
        
            features = extract_features(audio_path)
         
            data.append(features)
            labels.append(label)  
            
    return data, labels



def dataset_making(folder_path):


    """
    
    this takes the main folder where subfolders of audio exist and pass that through process_audio_folder function and save the labels
    and data into a csv format you can implement database for this (optional)
    
    """



    all_data=[]
    for root,dirs,files in os.walk(folder_path):
        if root==folder_path:
            continue

        sub_path = root
        lab=root.split("/")[1]

    
        data, labels = process_audio_folder(sub_path, label=lab)

        df = pd.DataFrame(data)


        df['label'] = labels
        
        all_data.append(df)
        print(f"finished {lab}") # remove this if you find this annoying
    result=pd.concat(all_data,axis=0)
    result.to_csv('extracted_data/audio_features.csv', index=False)

    print("CSV file has been created with audio features.") 
        

if __name__=="__main__":
    dataset_making("Audio")


