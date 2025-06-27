from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
import io

from app.services.prediction import PredictionService

router = APIRouter()
predictor = PredictionService()


@router.post("/predict_json")
def predict_json(data: list[dict]):
    try:
        return predictor.predict_from_json(data)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/predict_file")
async def predict_file(file: UploadFile = File(...)):
    if not file.filename.endswith(".csv"):
        raise HTTPException(status_code=400, detail="Файл должен быть в формате .csv")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))
        predictions = predictor.predict_from_file(df)
        return JSONResponse(content={"predictions": predictions})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ошибка обработки файла: {e}")
