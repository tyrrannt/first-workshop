import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException

from app.validators import validate_input_data
from app.config import FEATURES_PATH, MODEL_PATH, THRESHOLD_PATH

# === Загрузка обученных объектов ===
try:
    MODEL = joblib.load(MODEL_PATH)
    THRESHOLD = joblib.load(THRESHOLD_PATH)
    FEATURES = joblib.load(FEATURES_PATH)
except FileNotFoundError:
    print("Некоторые артефакты отсутствуют — проверь, сохранены ли они")


def min_max_normalize(df, column, min_val, max_val) -> pd.DataFrame:
    """
    Нормализует указанную колонку DataFrame с использованием метода Min-Max.

    Формула нормализации:
        (X - X_min) / (X_max - X_min)

    Parameters
    ----------
    df : pandas.DataFrame
        Входной DataFrame, содержащий данные.
    column : str
        Название колонки, которую необходимо нормализовать.
    min_val : float or int
        Минимальное значение для нормализации.
    max_val : float or int
        Максимальное значение для нормализации.

    Returns
    -------
    pandas.DataFrame
        DataFrame с обновлённой колонкой, значения которой находятся в диапазоне [0, 1].
    """
    df[column] = (df[column] - min_val) / (max_val - min_val)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняет предварительную обработку данных перед дальнейшей работой.

    Основные шаги:
    - Нормализует имена колонок, удаляя лишние пробелы и заменяя их на подчёркивания.
    - Удаляет строки с пропущенными значениями.
    - Нормализует значения в выбранных колонках с помощью метода Min-Max.
    - Преобразует категориальный признак 'Gender' в численный (Male=1, Female=0).
    - Округляет и конвертирует значения в определённых столбцах к типу int64.

    Parameters
    ----------
    df : pandas.DataFrame
        Входной DataFrame, содержащий сырые данные для предобработки.

    Returns
    -------
    pandas.DataFrame
        Обработанный DataFrame, готовый к дальнейшему анализу или моделированию.
    """
    df.columns = df.columns.str.strip().str.replace(" ", "_").str.replace("-", "_")
    df.dropna(inplace=True)

    # Применяем к каждому из двух столбцов
    df = min_max_normalize(df, 'Stress_Level', min_val=0, max_val=10)
    df = min_max_normalize(df, 'Physical_Activity_Days_Per_Week', min_val=0, max_val=7)

    # Преобразование Gender
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"Male": 1, "Female": 0})

    # Список столбцов, которые должны быть целыми
    columns_to_int = ['Family_History', 'Smoking', 'Diabetes', 'Obesity', 'Gender', 'Diet',
                      'Previous_Heart_Problems', 'Medication_Use', 'Alcohol_Consumption']

    # Округляем и конвертируем
    for col in columns_to_int:
        if col in df.columns:
            df[col] = df[col].round().astype('int64')
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Выполняет инженерию признаков на основе исходного DataFrame.

    Создаёт новые признаки, основанные на комбинациях и преобразованиях существующих столбцов.
    Функция возвращает копию исходного DataFrame с добавленными новыми признаками.

    Args:
        df (pd.DataFrame): Входной DataFrame, содержащий исходные данные.

    Returns:
        pd.DataFrame: DataFrame с дополнительными столбцами:
            - Cholesterol_Triglycerides: произведение значений холестерина и триглицеридов.
            - Blood_Pressure_Mean: среднее значение между систолическим и диастолическим давлением.
            - Exercise_Activity: суммарная физическая активность, рассчитанная как произведение часов
                                занятий в неделю и дней физической активности в неделю.
    """
    if not all(col in df.columns for col in
               ["Cholesterol", "Triglycerides", "Systolic_blood_pressure", "Diastolic_blood_pressure",
                "Exercise_Hours_Per_Week", "Physical_Activity_Days_Per_Week"]):
        raise ValueError("Отсутствуют необходимые столбцы для инженерии функций")

    X = df.copy()
    # Полиномиальные/произвольные фичи
    X["Cholesterol_Triglycerides"] = X["Cholesterol"] * X["Triglycerides"]
    X["Blood_Pressure_Mean"] = (X["Systolic_blood_pressure"] + X["Diastolic_blood_pressure"]) / 2
    X["Exercise_Activity"] = X["Exercise_Hours_Per_Week"] * X["Physical_Activity_Days_Per_Week"]

    return X


def validate_data(df: pd.DataFrame):
    """
    Выполняет валидацию входных данных перед дальнейшей обработкой.

    Эта функция вызывает `validate_input_data`, чтобы проверить структуру и содержимое DataFrame.
    В случае ошибки валидации выбрасывается исключение HTTPException со статус-кодом 400 и описанием проблемы.

    Args:
        df (pd.DataFrame): Входной DataFrame, который требуется проверить.

    Raises:
        HTTPException: Если данные не прошли валидацию. Статус-код 400, детализация ошибки.
    """
    # Валидация
    try:
        validate_input_data(df)
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=str(ve))


def prepare_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Подготавливает данные для моделирования.

    Эта функция последовательно выполняет предварительную обработку данных,
    инженерию признаков, а затем извлекает признаки из DataFrame.

    Args:
        df (pandas.DataFrame): Исходный DataFrame, содержащий сырые данные.

    Returns:
        pandas.DataFrame: DataFrame, содержащий только признаки, выбранные для моделирования.
    """
    df = preprocess(df)
    df = feature_engineering(df)
    X = df[FEATURES]
    return X


def predict(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Выполняет предсказание вероятностей и классов для заданных признаков.

    Эта функция использует обученную модель (MODEL) для получения вероятностей
    принадлежности к положительному классу, а затем преобразует эти вероятности в
    бинарные предсказания на основе заданного порога (THRESHOLD).

    Args:
        X (pandas.DataFrame): DataFrame с признаками, для которых нужно выполнить предсказание.

    Returns:
        tuple: Кортеж из двух массивов:
            - probs (numpy.ndarray): Массив вероятностей положительного класса.
            - preds (numpy.ndarray): Массив бинарных предсказаний (0 или 1).
    """
    probs = MODEL.predict_proba(df)[:, 1]
    preds = (probs > THRESHOLD).astype(int)
    return probs, preds
