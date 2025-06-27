from sklearn.metrics import f1_score

from app.data_loader import load_dataset
from app.model_trainer.save_load import save_model, save_json
from app.preprocessing import preprocess, feature_engineering
from app.model_trainer.pipelines import get_models, get_param_grids, get_preprocessor
from app.model_trainer.trainer import ModelTrainer
from app.model_trainer.evaluator import ModelEvaluator
from app.config import TARGET, MEDICAL_FEATURES, THRESHOLD_PATH, MODEL_PATH, CATEGORICAL_FEATURES, FEATURES_PATH, \
    METRICS_PATH

df = load_dataset()
df = preprocess(df)
df = feature_engineering(df)
df[TARGET] = df[TARGET].round().astype("int64")

# Определение признаков
features = df.corr()[TARGET].abs().sort_values(ascending=False)
selected = features[features >= 0.015].index.tolist()
X = df[list(set(selected + MEDICAL_FEATURES))].drop(columns=[TARGET])
y = df[TARGET]

categorical = X.select_dtypes(include='int64').columns.tolist()
preprocessor = get_preprocessor(categorical)
trainer = ModelTrainer(X, y, preprocessor, get_models(), get_param_grids())
results = trainer.train_all()

# Поиск лучшей модели
best_model_name = max(results, key=lambda k: f1_score(
    results[k]["y_val"],
    (results[k]["best_model"].predict_proba(results[k]["X_val"])[:, 1] > 0.5).astype(int)
))
best = results[best_model_name]
evaluator = ModelEvaluator(best["best_model"], best["X_val"], best["y_val"])
threshold = evaluator.find_best_threshold()
metrics = evaluator.evaluate(threshold)
metrics["best_params"] = best["best_params"]

# Сохранение
save_model(best["best_model"], MODEL_PATH)
save_model(threshold, THRESHOLD_PATH)
save_model(X.columns.tolist(), FEATURES_PATH)
save_model(categorical, CATEGORICAL_FEATURES)

save_json(metrics, METRICS_PATH)

print(f"Лучшая модель: {best_model_name}, F1: {metrics['f1_score']:.4f}")
