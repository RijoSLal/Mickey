import os 

os.environ["CUDA_VISIBLE_DEVICES"]="-1" # if you want to run this in GPU change -1 to 1 or 2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from tensorflow import keras 
from keras.layers import Dense, LSTM, GRU
import joblib
from sklearn.model_selection import train_test_split
import subprocess

class MusicModel:
    def __init__(self, input_shape: tuple, path_to_dataset: str):
        self.input_shape = input_shape
        self.path_to_dataset = path_to_dataset

    def classification_model(self):
        
        """
        this is the structure of model
        """

        inputs = keras.Input(shape = self.input_shape)
        lstm = LSTM(256, activation = "relu", return_sequences = True, dropout = 0.2)(inputs)
        gru = GRU(128, activation = "relu", dropout = 0.2)(lstm)
        linear = Dense(64, activation = "relu")(gru)
        outputs = Dense(10)(linear)
        model = keras.Model(inputs = inputs, outputs = outputs)
        return model


    def preprocessing_pipeline(self):

        """
        model is preprocessed and scaling variable is saved
        
        """

        dataset = pd.read_csv(self.path_to_dataset)
        datasets = dataset.drop(["label"], axis=1)

        std_scaler = StandardScaler()
        scaled = std_scaler.fit_transform(datasets)

        X = np.reshape(scaled, (scaled.shape[0], 1, scaled.shape[1]))
        Y = dataset["label"].map({
                'sad': 0,
                'serenity': 1,
                'nostalgia': 2,
                'joy': 3,
                'confidence': 4,
                'relax': 5,
                'angry': 6,
                'happy': 7,
                'chill': 8,
                'excitement': 9
        })

        joblib.dump(std_scaler, 'model_scaler/scaler.pkl')
        return X, Y



    def fit_model(self, X, Y, epo = 50, batch = 32):

        """
        building and evaluating model

        """
        log_dir = "logs"
        tensorboard_callback = keras.callbacks.TensorBoard(
            log_dir = log_dir,
            histogram_freq = 1
        )
        
        model = self.classification_model()
        model.compile(
            optimizer = keras.optimizers.Adam(learning_rate=0.001),
            loss = keras.losses.SparseCategoricalCrossentropy(from_logits=True),
            metrics = ['accuracy']
        )
        x_train, x_test, y_train, y_test = train_test_split(X, Y, random_state = 24, shuffle = True, test_size = 0.1)

        model.fit(
            x_train, 
            y_train, 
            epochs = epo, 
            batch_size = batch, 
            callbacks=[tensorboard_callback]
        )

        loss, accuracy = model.evaluate(x_test, y_test)
        
        save_to_path = 'model_scaler/music_model.h5'
        model.save(save_to_path)

        from io import StringIO

        stream = StringIO()
        model.summary(print_fn = lambda x: stream.write(x + "\n"))
        summary = stream.getvalue()
        f"""
            +----------------------------------------------+
             Loss : {loss} , Accuracy : {accuracy}       
                                                          
             Model Summary:                               
                  {summary}
                                                          
             Model Successfully Saved: {save_to_path}                                        
            +----------------------------------------------+
        """
        
        return None
    
    def model_evaluation(self):
        subprocess.run(["tensorboard --logdir logs"])



if __name__=="__main__":

    music_model = MusicModel(
        input_shape = (1,57), 
        path_to_dataset = 'extracted_data/audio_features.csv'
    )
    x, y = music_model.preprocessing_pipeline()
    music_model.fit_model(x, y)