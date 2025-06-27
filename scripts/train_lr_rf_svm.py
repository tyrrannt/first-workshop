import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
import json
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, roc_auc_score, confusion_matrix, classification_report
import joblib
import os
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC

from app.config import TARGET, MEDICAL_FEATURES, TRAIN_PATH
from app.preprocessing import preprocess, feature_engineering

# === 1. Загрузка данных и обработка ===

df = pd.read_csv(TRAIN_PATH)
df.drop(['Unnamed: 0', 'id'], axis=1, inplace=True)
df = preprocess(df)
df['Heart_Attack_Risk_(Binary)'] = df['Heart_Attack_Risk_(Binary)'].round().astype('int64')

df = feature_engineering(df)

df_encoded = df.copy()

# Вычислим матрицу корреляций
corr = df_encoded.corr()

# Отсортированный список корреляций с целевой переменной
correlations_with_target = corr['Heart_Attack_Risk_(Binary)'].sort_values(ascending=False)

threshold = 0.015
important_features = correlations_with_target[abs(correlations_with_target) >= threshold].index.tolist()
# === Разделение на признаки и целевую переменную ===
y = df[TARGET]
FEATURE = list(set(important_features + MEDICAL_FEATURES))
# === Признаки ===
FEATURES = df[FEATURE].columns.drop(TARGET).tolist()

# === Категориальные признаки (int64, но не целевая) ===
categorical_features = df[FEATURES].select_dtypes(include='int64').columns.tolist()

# === Предобработка категориальных признаков: кодирование ===
categorical_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))  # Для новых значений в тесте
])

# === Полный процессор данных ===
preprocessor = ColumnTransformer([
    ('cat', categorical_pipeline, categorical_features)
], remainder='passthrough')

# === Модель ===
models = {
    "RandomForest": RandomForestClassifier(random_state=42),
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
    "SVM": SVC(probability=True, random_state=42)
}

# === Сетки параметров ===
param_grids = {
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
        'classifier__kernel': ['rbf', ],
        'classifier__gamma': ['scale', 'auto', 0.1, 1],
        'classifier__degree': [2, 3],
        'classifier__class_weight': ['balanced']
    }
}

# === Разделение данных ===
X = df[FEATURES]
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Так как мы видем дизбаланс классов, то применяем SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = 'f1'

results = {}

for model_name, model in models.items():

    print(f"\n{'=' * 40}\nОбучение модели: {model_name}\n{'=' * 40}")

    # === Пайплайн с предобработкой + модель ===
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # === Поиск гиперпараметров ===
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grids[model_name],
        scoring=scoring,
        cv=cv,
        n_jobs=-1,
        verbose=1
    )

    grid.fit(X_res, y_res)

    best_model = grid.best_estimator_
    probs = best_model.predict_proba(X_val)[:, 1]

    # === Подбор порога ===
    best_threshold = 0.5
    best_f1 = 0
    for t in np.arange(0.1, 0.9, 0.01):
        preds = (probs > t).astype(int)
        score = f1_score(y_val, preds, zero_division=0)
        if score > best_f1:
            best_f1 = score
            best_threshold = t

    final_preds = (probs > best_threshold).astype(int)

    # === Метрики ===
    report = classification_report(y_val, final_preds, output_dict=True, zero_division=0)
    auc = roc_auc_score(y_val, probs)

    results[model_name] = {
        "best_model": best_model,
        "best_params": grid.best_params_,
        "best_threshold": best_threshold,
        "best_f1": best_f1,
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "roc_auc": auc,
        "pipeline": grid.best_estimator_
    }

best_model_name = max(results, key=lambda k: results[k]["best_f1"])
best = results[best_model_name]

# === Предсказание вероятностей на валидации ===
probs = best['best_model'].predict_proba(X_val)[:, 1]

# === Поиск лучшего порога ===
best_threshold = 0.5
best_f1 = 0
for t in np.arange(0.1, 0.9, 0.01):
    preds = (probs > t).astype(int)
    score = f1_score(y_val, preds)
    if score > best_f1:
        best_f1 = score
        best_threshold = t

# === Финальная метрика ===
final_preds = (probs > best_threshold).astype(int)
report = classification_report(y_val, final_preds, output_dict=True, zero_division=0)
cm = confusion_matrix(y_val, final_preds)

# === Сохранение артефактов ===
os.makedirs("../models", exist_ok=True)
joblib.dump(best['best_model'], "../models/best_model.pkl")
joblib.dump(best_threshold, "../models/best_threshold.pkl")
joblib.dump(FEATURES, "../models/features.pkl")
joblib.dump(best['pipeline'], "../models/pipeline.pkl")
joblib.dump(smote, "../models/smote.pkl")
joblib.dump(categorical_features, "../models/categorical_features.pkl")

print("Модель, и порог сохранены в папке 'models'.")

# === Сохранение метрик ===
metrics = {
    "best_threshold": float(best_threshold),
    "f1_score": report["1"]["f1-score"],
    "precision": report["1"]["precision"],
    "recall": report["1"]["recall"],
    "roc_auc": roc_auc_score(y_val, probs),
    "confusion_matrix": cm.tolist(),
    "classification_report": report,
    "best_params": best['best_params']
}

with open("../models/metrics.json", "w") as f:
    json.dump(metrics, f, indent=4)
