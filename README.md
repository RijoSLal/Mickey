# Mickey: Emotion Recognition in Music

Mickey is a ML based web app built to analyze and capture emotions in music using deep learning techniques. The model is powered by LSTM and GRU-based neural 
networks, and the application is deployed using FastAPI. It uses Librosa for audio processing and feature extraction, with Pandas and Numpy handling data manipulation.

## Features

- **Emotion Detection**: Detect emotions in music tracks using deep learning models.
- **LSTM & GRU**: The emotion recognition model is built with Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) for effective time series analysis.
- **FastAPI**: FastAPI is used for building the backend, providing high performance and easy API endpoints.
- **Audio Processing**: Librosa is used for feature extraction from audio tracks, including Mel spectrograms, chroma features, and more.
- **User Interface**: Jinja templates power the front-end, offering a simple yet efficient user interface for interacting with the application.

## Requirements

To run the project, you'll need the following libraries:

- **Pandas**: For handling data in tabular form.
- **NumPy**: For numerical operations and data manipulation.
- **TensorFlow**: For building and training the LSTM and GRU-based neural network models.
- **Librosa**: For audio processing and extracting features from music tracks.
- **FastAPI**: For backend development and creating RESTful APIs.
- **Jinja**: For rendering HTML templates.
- **Uvicorn**: ASGI server for FastAPI.

You can install the required dependencies using the following command:

```bash
pip install pandas numpy tensorflow librosa fastapi jinja2 uvicorn
```

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/RijoSLal/mickey.git
   ```
   
2. Navigate into the project directory:
   ```bash
   cd mickey
   ```

4. Run the FastAPI app:
   ```bash
   uvicorn fast:app --reload
   ```

   The application will be available at `http://127.0.0.1:8000`.

## Usage

1. **Upload a music file**: Navigate to the app's web interface and record an audio and submit.
2. **Get emotion prediction**: The model will process the audio file and provide an emotion classification based on the music's features and shows audio spectrogram.


## Contributing

If you'd like to contribute to Mickey, feel free to fork the repository and submit a pull request. Here are a few ways you can help:
- Report issues or bugs.
- Add new features or improve existing ones.
- Improve documentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

Feel free to reach out if you have any questions or suggestions!
