from sklearn.model_selection import GridSearchCV, StratifiedKFold
from imblearn.over_sampling import SMOTE
from sklearn.pipeline import Pipeline
import numpy as np

class ModelTrainer:
    def __init__(self, X, y, preprocessor, models, param_grids):
        self.X = X
        self.y = y
        self.preprocessor = preprocessor
        self.models = models
        self.param_grids = param_grids
        self.cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    def train_all(self, scoring="f1"):
        from sklearn.model_selection import train_test_split
        X_train, X_val, y_train, y_val = train_test_split(self.X, self.y, test_size=0.2, stratify=self.y, random_state=42)

        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)

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
            grid.fit(X_res, y_res)
            results[name] = {
                "best_model": grid.best_estimator_,
                "best_params": grid.best_params_,
                "X_val": X_val,
                "y_val": y_val
            }

        return results
