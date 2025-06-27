import pandas as pd

from app.config import TEST_PATH, RESULT_PATH
from app.model_trainer.save_load import save_csv
from app.preprocessing import predict, prepare_data

# === Считываем тестовые данные из CSV-файла, указанного в константе `TEST_PATH`. ===
df_test = pd.read_csv(TEST_PATH)
X_test = prepare_data(df_test)

# === Выполняем предсказание с помощью обученной модели через функцию `predict`. ===
probs, preds = predict(X_test)

# === Сохранение результата ===
df_result = pd.DataFrame({
    "id": df_test["id"],
    "prediction": preds,
    "probability": probs
})

save_csv(df_result, RESULT_PATH)
print("Файл 'results/predictions.csv' готов. Содержит столбцы: id и prediction и probability.")
