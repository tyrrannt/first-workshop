import pandas as pd
from app.config import TRAIN_PATH

def load_dataset():
    df = pd.read_csv(TRAIN_PATH)
    df.drop(['Unnamed: 0', 'id'], axis=1, inplace=True)
    return df
