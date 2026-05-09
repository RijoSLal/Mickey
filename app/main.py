from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import sys
from pathlib import Path

# Add the app directory to sys.path to allow absolute imports of local modules
app_dir = Path(__file__).resolve().parent
if str(app_dir) not in sys.path:
    sys.path.append(str(app_dir))

from model_inference import model_prediction
from fastapi.staticfiles import StaticFiles

app = FastAPI()

BASE_DIR = app_dir.parent

app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

import tempfile
import os

@app.post("/process_audio/")
async def process_audio(request: Request, file: UploadFile = File(...)) -> HTMLResponse:

    """
    process uploaded audio file and return emotion and spectrogram image

    steps
    save uploaded file to a temporary file (deleted automatically after use)
    run emotion model on audio data
    load audio and create spectrogram using librosa
    convert spectrogram to image and encode as base64
    return html template with image and emotion

    args:
        request: fastapi request object for template rendering
        file: uploaded audio file

    returns:
        html template with spectrogram image and predicted emotion

    """
    
    suffix = Path(file.filename).suffix if file.filename else ".wav"
    
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Predict emotion
        emo = model_prediction(tmp_path)

        # Load audio for spectrogram
        y, _ = librosa.load(tmp_path)

        plt.figure(figsize=(10, 4))
        DB_SCALED = librosa.amplitude_to_db(
            np.abs(librosa.stft(y)),
            ref=np.max
        )

        plt.imshow(DB_SCALED, aspect="auto", origin="lower", cmap='BuPu')
        plt.colorbar(format="%+2.0f dB")

        img_buf = io.BytesIO()
        plt.savefig(img_buf, format="png")
        img_buf.seek(0)
        img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')
        plt.close()
   
        return templates.TemplateResponse(
            "mickey.html", {"request": request, "img_base64": img_base64 ,"emotion":emo}
        )
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    """
    show home page with no processed data

    args:
        request: fastapi request object

    returns
        html template
    """
    return templates.TemplateResponse(
        "mickey.html", 
        {
            "request": request, 
            "img_base64": None
        }
    )









