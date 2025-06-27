import pandas as pd
from app.config import TRAIN_PATH


def load_dataset():
    """
    Загружает обучающий набор данных из CSV-файла.

    Функция считывает данные из файла, указанного в константе `TRAIN_PATH`, и удаляет
    неиспользуемые колонки 'Unnamed: 0' и 'id'.

    Returns
    -------
    pandas.DataFrame
        DataFrame с данными из CSV-файла без указанных колонок.
    """
    df = pd.read_csv(TRAIN_PATH)
    df.drop(['Unnamed: 0', 'id'], axis=1, inplace=True)
    return df
