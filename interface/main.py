from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from routes import upload, processing, visualization

app = FastAPI()

# Подключаем маршруты
app.include_router(upload.router)
app.include_router(processing.router)
app.include_router(visualization.router)

# Подключаем статику
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)