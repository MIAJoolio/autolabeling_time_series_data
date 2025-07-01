from fastapi import APIRouter, Request, Form
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
import os
import numpy as np
from utils.clustering import create_experiment
from utils.mlflow_logger import setup_mlflow

router = APIRouter()
templates = Jinja2Templates(directory="templates")

@router.get("/processing", response_class=HTMLResponse)
async def processing_page(request: Request):
    return templates.TemplateResponse("processing.html", {"request": request})

@router.post("/run-clustering")
async def run_clustering(
    fe_method: str = Form(...),
    model_name: str = Form(...),
    fs: int = Form(500),
    n_freqs: int = Form(10),
    period: int = Form(7),
    order: int = Form(1),
    n_clusters: int = Form(2),
    eps: float = Form(0.5),
    min_samples: int = Form(5),
    distance_threshold: float = Form(0.5),
    metric: str = Form("euclidean")
):
    try:
        X_train = np.load("data/data.npy")
    except FileNotFoundError:
        return {"error": "No data uploaded yet"}

    params = {
        "fs": fs,
        "n_freqs": n_freqs,
        "period": period,
        "order": order,
        "n_clusters": n_clusters,
        "eps": eps,
        "min_samples": min_samples,
        "distance_threshold": distance_threshold,
        "metric": metric
    }

    setup_mlflow("ts_clustering_exp")
    create_experiment("ts_clustering_exp", [params], fe_method, X_train)

    return RedirectResponse(url="/visualization/", status_code=303)