# app/validators.py
import pandas as pd
from .schemas import Record
from pydantic import ValidationError


def validate_input_data(df: pd.DataFrame) -> None:
    """
    Проверяет данные DataFrame на соответствие модели Record.

    Args:
        df (pd.DataFrame): Входной DataFrame, содержащий записи для валидации.

    Raises:
        ValueError: Если валидация какой-либо записи завершается ошибкой.
            Сообщение содержит номер некорректной записи и причину ошибки.
    """
    records = df.to_dict(orient="records")
    try:
        for i, record in enumerate(records):
            Record(**record)
    except ValidationError as e:
        raise ValueError(f"Ошибка валидации записи {i}: {e}")
