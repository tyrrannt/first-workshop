import pandas as pd
from app.preprocessing import validate_data, prepare_data, predict


class PredictionService:
    def __init__(self):
        pass

    def predict_from_json(self, data: list[dict]) -> list[dict]:
        df = pd.DataFrame(data)
        X = prepare_data(df)
        validate_data(df)
        probs, preds = predict(X)

        df["prediction"] = preds
        df["probability"] = probs.round(4)

        return df[["id", "prediction", "probability"]].to_dict(orient="records")

    def predict_from_file(self, df: pd.DataFrame) -> list[dict]:
        X = prepare_data(df)
        validate_data(df)
        probs, preds = predict(X)

        result = [
            {"id": int(row["id"]), "prediction": int(pred)}
            for row, pred in zip(df.to_dict(orient="records"), preds)
        ]
        return result
