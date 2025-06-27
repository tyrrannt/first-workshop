from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def get_preprocessor(categorical_features):
    """
    Создает и возвращает ColumnTransformer для предварительной обработки категориальных признаков.

    Категориальные признаки кодируются с помощью OneHotEncoder, остальные признаки проходят без изменений.

    Parameters
    ----------
    categorical_features : list of str
        Список названий категориальных признаков, которые необходимо закодировать.

    Returns
    -------
    ColumnTransformer
        Объект ColumnTransformer, готовый к использованию для преобразования данных.
    """
    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    return ColumnTransformer([
        ('cat', categorical_pipeline, categorical_features)
    ], remainder='passthrough')



def get_models():
    """
    Возвращает словарь с экземплярами классификаторов машинного обучения.

    В данный момент поддерживаются следующие модели:
    - RandomForest: классификатор на основе случайного леса.
    - LogisticRegression: логистическая регрессия.
    - SVM: машина опорных векторов с поддержкой вероятностной оценки.

    Returns
    -------
    dict
        Словарь, где ключ — имя модели, значение — объект классификатора.
    """
    return {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(probability=True, random_state=42)
    }



def get_param_grids():
    """
    Возвращает словарь с сетками гиперпараметров для различных моделей классификации.

    Каждая модель связана со своей сеткой параметров, используемой в процессе подбора гиперпараметров
    (например, с помощью GridSearchCV). Сетки включают различные значения для настройки производительности модели.

    Returns
    -------
    dict
        Словарь, где ключ — имя модели, значение — словарь с параметрами и списком их возможных значений.
        Поддерживаемые модели:
        - "RandomForest": случайный лес.
        - "LogisticRegression": логистическая регрессия.
        - "SVM": машина опорных векторов.

    Notes
    -----
    Имена параметров включают префикс 'classifier__', так как модели обычно используются внутри Pipeline,
    где классификатор находится в шаге с именем 'classifier'.
    """
    return {
        "RandomForest": {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [10, 15, 20, None],
            'classifier__min_samples_split': [2, 5],
            'classifier__min_samples_leaf': [1, 2],
        },
        "LogisticRegression": {
            'classifier__C': [0.01, 0.1, 1, 10],
            'classifier__penalty': ['l1', 'l2'],
            'classifier__solver': ['liblinear']
        },
        "SVM": {
            'classifier__C': [0.1, 1, 10],
            'classifier__kernel': ['rbf'],
            'classifier__gamma': ['scale', 'auto', 0.1, 1],
            'classifier__degree': [2, 3],
            'classifier__class_weight': ['balanced']
        }
    }

