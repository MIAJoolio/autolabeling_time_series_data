from fastapi import APIRouter, Request
from fastapi.templating import Jinja2Templates
import os

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/visualization", response_class=HTMLResponse)
async def visualization_page(request: Request):
    images_dir = "static/images"
    os.makedirs(images_dir, exist_ok=True)
    images = os.listdir(images_dir)
    return templates.TemplateResponse("visualization.html", {"request": request, "images": images})