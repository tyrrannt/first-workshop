import os
import json
import joblib
from typing import Any

from pandas.core.interchange.dataframe_protocol import DataFrame


def save_model(model: Any, path: str) -> None:
    """
    Сохраняет модель или любой объект joblib в указанный путь.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
    print(f"Сохранено: {path}")


def load_model(path: str) -> Any:
    """
    Загружает модель из указанного пути.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    return joblib.load(path)


def save_json(data: dict, path: str) -> None:
    """
    Сохраняет словарь как JSON-файл.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=4)
    print(f"JSON сохранён: {path}")


def load_json(path: str) -> dict:
    """
    Загружает JSON-файл и возвращает как словарь.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Файл не найден: {path}")
    with open(path, "r") as f:
        return json.load(f)


def save_csv(data: DataFrame, path: str) -> None:
    """
    Сохраняет DataFrame в CSV-файл.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    data.to_csv(path, index=False)
    print(f"CSV сохранён: {path}")
