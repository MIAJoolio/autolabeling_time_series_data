from fastapi import APIRouter, Request, UploadFile, File
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import os
import numpy as np
import pandas as pd

router = APIRouter()
templates = Jinja2Templates(directory="templates")
DATA_DIR = "data"

os.makedirs(DATA_DIR, exist_ok=True)

@router.get("/", response_class=HTMLResponse)
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@router.post("/upload/")
async def handle_upload(file: UploadFile = File(...)):
    file_path = os.path.join(DATA_DIR, file.filename)

    if file.filename.endswith(".csv"):
        df = pd.read_csv(file.file)
        data = df.values
    elif file.filename.endswith(".npy"):
        data = np.load(file.file)
    else:
        return {"error": "Unsupported format"}

    np.save(file_path.replace('.csv', '.npy').replace('.npy', '_data.npy'), data)

    return RedirectResponse(url="/processing/", status_code=303)