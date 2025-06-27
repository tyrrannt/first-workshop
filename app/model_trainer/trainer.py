from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline


class ModelTrainer:
    def __init__(self, X, y, preprocessor, models, param_grids):
        """
        Инициализирует объект ModelTrainer для обучения моделей машинного обучения.

        Parameters
        ----------
        X : array-like
            Массив признаков.
        y : array-like
            Массив целевых меток.
        preprocessor : ColumnTransformer
            Объект ColumnTransformer для предварительной обработки данных.
        models : dict
            Словарь с именами моделей в качестве ключей и экземплярами классификаторов в качестве значений.
        param_grids : dict
            Словарь с именами моделей в качестве ключей и сетками гиперпараметров в качестве значений.
        """
        self.X = X
        self.y = y
        self.preprocessor = preprocessor
        self.models = models
        self.param_grids = param_grids
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def train_all(self, scoring="f1"):
        """
        Обучает все модели из словаря `models` с использованием кросс-валидации и подбора гиперпараметров.

        Данные разделяются на обучающую и валидационную выборки. Обучающая выборка балансируется с помощью SMOTE.
        Для каждой модели строится pipeline, включающий предобработку и саму модель. Подбираются оптимальные
        гиперпараметры с использованием GridSearchCV. Результаты сохраняются в виде словаря.

        Parameters
        ----------
        scoring : str, optional
            Метрика оценки качества модели. По умолчанию "f1".

        Returns
        -------
        dict
            Словарь с результатами обучения. Каждый элемент соответствует одной модели и содержит:
            - best_model: лучшая модель после подбора гиперпараметров.
            - best_params: лучшие гиперпараметры.
            - X_val: валидационная выборка признаков.
            - y_val: валидационная выборка целевых меток.
        """
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, stratify=self.y,
                                                          random_state=42)
        #
        # smote = SMOTE(random_state=42)
        # X_res, y_res = smote.fit_resample(X_train, y_train)

        results = {}

        for name, model in self.models.items():
            pipeline = Pipeline([
                ('preprocessor', self.preprocessor),
                ('classifier', model)
            ])

            grid = GridSearchCV(
                pipeline,
                self.param_grids[name],
                scoring=scoring,
                cv=self.cv,
                n_jobs=-1,
                verbose=1
            )
            # grid.fit(X_res, y_res)
            grid.fit(X_train, y_train)
            results[name] = {
                "best_model": grid.best_estimator_,
                "best_params": grid.best_params_,
                "X_val": X_val,
                "y_val": y_val
            }

        return results
