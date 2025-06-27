from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io

from app.services.prediction import PredictionService

router = APIRouter()
predictor = PredictionService()


@router.post("/predict_json")
def predict_json(data: list[dict]):
    """
    Обрабатывает POST-запрос с данными в формате JSON для получения предсказаний модели.

    Эта функция принимает список словарей, представляет его как структурированные данные,
    и передаёт на обработку сервису `PredictionService`. Если произошла ошибка,
    возвращает HTTPException со статус-кодом 400 и описанием ошибки.

    Parameters
    ----------
    data : list[dict]
        Список словарей, где каждый словарь представляет запись с признаками.

    Returns
    -------
    dict
        Результат работы метода `predict_from_json` из класса `PredictionService`.

    Raises
    ------
    HTTPException
        Статус 400, если произошла ошибка при обработке входных данных.
    """
    try:
        return predictor.predict_from_json(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    """
    Обрабатывает POST-запрос с загрузкой CSV-файла для получения предсказаний модели.

    Эта асинхронная функция проверяет, что загруженный файл имеет расширение .csv,
    считывает его содержимое, преобразует в DataFrame и передаёт на обработку сервису `PredictionService`.
    Возвращает результат в виде JSON-ответа. При возникновении ошибки — выбрасывает исключение.

    Parameters
    ----------
    file : UploadFile
        Загружаемый файл. Должен быть в формате CSV.

    Returns
    -------
    JSONResponse
        Ответ в формате JSON, содержащий список предсказаний.

    Raises
    ------
    HTTPException
        - Статус 400, если загруженный файл не является CSV.
        - Статус 500, если произошла ошибка при обработке файла.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Файл должен быть в формате .csv")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        predictions = predictor.predict_from_file(df)
        return JSONResponse(content={"predictions": predictions})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {e}")
