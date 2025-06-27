"""
Модуль запуска FastAPI-приложения для предсказания рисков сердечного приступа.
"""

from fastapi import FastAPI
import uvicorn

from app.config import HOST, PORT
from app.api import router as prediction_router

app = FastAPI(title="Предсказание рисков сердечного приступа")
app.include_router(prediction_router)

if __name__ == "__main__":
    """
    Запускает локальный сервер приложения.

    Используется uvicorn для запуска FastAPI-приложения. Указаны параметры:
    - host: адрес хоста (из конфигурации).
    - port: порт (из конфигурации).
    - reload: автоматическая перезагрузка сервера при изменении кода.
    """
    uvicorn.run("app.main:app", host=HOST, port=PORT, reload=True)
