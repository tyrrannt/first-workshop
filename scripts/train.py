import os, json, joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score

from imblearn.over_sampling import SMOTE

from app.config import TARGET, MEDICAL_FEATURES, TRAIN_PATH, THRESHOLD_PATH, FEATURES_PATH, PIPELINE_PATH
from app.preprocessing import preprocess, feature_engineering

# === Загрузка и подготовка данных ===
df = pd.read_csv(TRAIN_PATH).drop(['Unnamed: 0', 'id'], axis=1)
df = preprocess(df)
df[TARGET] = df[TARGET].round().astype(int)
df = feature_engineering(df)

# === Отбор признаков ===
corr = df.corr()[TARGET].abs().sort_values(ascending=False)
selected = corr[corr >= 0.015].index.tolist()
features = list(set(selected + MEDICAL_FEATURES) - {TARGET})

X, y = df[features], df[TARGET]
num_features = X.select_dtypes(include='float64').columns.tolist()
cat_features = X.select_dtypes(include='int64').columns.tolist()

# === Обучение модели ===
X_train, X_val, y_train, y_val = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# === Преобразования ===
categorical_pipeline = Pipeline([
    ('encoder', OneHotEncoder(handle_unknown='ignore'))
])

preprocessor = ColumnTransformer([
    ('cat', categorical_pipeline, cat_features)
], remainder='passthrough')  # Числовые передаём как есть

# Фиксированная модель с уже найденными лучшими параметрами
model = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=42
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', model)
])

# Обучение
pipeline.fit(X_train, y_train)

# Предсказание вероятностей
probs = pipeline.predict_proba(X_val)[:, 1]

# Поиск лучшего порога
best_threshold, best_f1 = 0.5, 0
for t in np.arange(0.1, 0.9, 0.01):
    preds = (probs > t).astype(int)
    score = f1_score(y_val, preds)
    if score > best_f1:
        best_f1, best_threshold = score, t

# Финальное предсказание
final_preds = (probs > best_threshold).astype(int)

# Метрики
report = classification_report(y_val, final_preds, output_dict=True, zero_division=0)
roc = roc_auc_score(y_val, probs)
cm = confusion_matrix(y_val, final_preds)

# Сохранение артефактов
os.makedirs("../models", exist_ok=True)
joblib.dump(best_threshold, THRESHOLD_PATH)
joblib.dump(list(X_train.columns), FEATURES_PATH)
joblib.dump(pipeline, PIPELINE_PATH)

with open("../models/metrics.json", "w") as f:
    json.dump({
        "best_threshold": float(best_threshold),
        "f1_score": report["1"]["f1-score"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "roc_auc": roc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report
    }, f, indent=4)

print("Готово. Модель и метрики сохранены.")
