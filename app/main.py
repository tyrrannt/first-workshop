from fastapi import FastAPI
import uvicorn

from app.config import HOST, PORT
from app.api import router as prediction_router

app = FastAPI(title="Предсказание рисков сердечного приступа")
app.include_router(prediction_router)

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
