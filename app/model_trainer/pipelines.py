from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def get_preprocessor(categorical_features):
    categorical_pipeline = Pipeline([
        ('encoder', OneHotEncoder(handle_unknown='ignore'))
    ])
    return ColumnTransformer([
        ('cat', categorical_pipeline, categorical_features)
    ], remainder='passthrough')


def get_models():
    return {
        "RandomForest": RandomForestClassifier(random_state=42),
        # "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        # "SVM": SVC(probability=True, random_state=42)
    }


def get_param_grids():
    return {
        "RandomForest": {
            'classifier__n_estimators': [100, 200],
            'classifier__max_depth': [15, 16, 17, 18, None],
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
