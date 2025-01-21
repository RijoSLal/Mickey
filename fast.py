from fastapi import FastAPI, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pathlib import Path
import librosa
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from model_run import model_prediction_with_temp
app = FastAPI()


templates = Jinja2Templates(directory="templates")


@app.post("/process_audio/")
async def process_audio(request: Request, file: UploadFile = File(...)):

    temp_file_path = Path(f"temp_{file.filename}")
    with open(temp_file_path, "wb") as buffer:
        buffer.write(await file.read())

    """
    convert audio into spectrogram
    
    
    """
    emo=model_prediction_with_temp(temp_file_path)
    y, sr = librosa.load(temp_file_path)
    plt.figure(figsize=(10, 4))
    D = librosa.amplitude_to_db(np.abs(librosa.stft(y)), ref=np.max)
    plt.imshow(D, aspect="auto", origin="lower", cmap='BuPu')
    plt.colorbar(format="%+2.0f dB")


    img_buf = io.BytesIO()
    plt.savefig(img_buf, format="png")
    img_buf.seek(0)
    img_base64 = base64.b64encode(img_buf.read()).decode('utf-8')

    
    temp_file_path.unlink()
   
    

    return templates.TemplateResponse(
        "index.html", {"request": request, "img_base64": img_base64 ,"emotion":emo}
    )


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "img_base64": None})









