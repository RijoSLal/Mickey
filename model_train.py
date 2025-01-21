import os 
os.environ["CUDA_VISIBLE_DEVICES"]="-1" #if you want to run this in GPU change -1 to 1 or 2
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import pandas as pd 
import numpy as np 
from sklearn.preprocessing import StandardScaler
from tensorflow import keras 
from keras.layers import Dense,LSTM,GRU
import joblib
from sklearn.model_selection import train_test_split



def create_model(input_shape):
    
    """
    this is the structure of model

    """

    inputs = keras.Input(shape=input_shape)
    x = LSTM(256, activation="relu", return_sequences=True, dropout=0.2)(inputs)
    x = GRU(128, activation="relu", dropout=0.2)(x)
    x = Dense(64, activation="relu")(x)
    outputs = Dense(10)(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model



def data_preperation(path_to_dataset:str):

    """
    model is preprocessed and scaling variable is saved
    
    """

    dataset=pd.read_csv(path_to_dataset)
    datasets=dataset.drop(["label"],axis=1)

    std_scaler=StandardScaler()
    scaled=std_scaler.fit_transform(datasets)

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

    return X,Y



def model_building_fit(X,Y,epoc=50,batch=32,input_shape = (1,57)):

    """
    building and evaluating model

    """

    
    model = create_model(input_shape)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
    x_train,x_test,y_train,y_test=train_test_split(X,Y,random_state=24,shuffle=True,test_size=0.1)
    model.fit(x_train, y_train, epochs=epoc, batch_size=batch)

    loss,acc=model.evaluate(x_test,y_test)
    print(f'loss : {loss} , accuracy : {acc}')

    model.save('model_scaler/music_model.h5')

    print("model saved successfully")

    return model.summary()



if __name__=="__main__":
   x,y = data_preperation('extracted_data/audio_features.csv')
   model_building_fit(x,y)