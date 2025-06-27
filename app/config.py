from pathlib import Path

# Путь относительно корня проекта
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # Поднимаемся на два уровня вверх
PKL_PATH = PROJECT_ROOT / "models"
DATA_PATH = PROJECT_ROOT / "data"
RESULT_PATH = PROJECT_ROOT / "results"

# === Настройки сервера ===
HOST = "127.0.0.1"
PORT = 8000

# === Пути к сохранённым моделям ===
MODEL_PATH = PKL_PATH / "best_model.pkl"
THRESHOLD_PATH = PKL_PATH / "best_threshold.pkl"
FEATURES_PATH = PKL_PATH / "features.pkl"
PIPELINE_PATH = PKL_PATH / "pipeline.pkl"
CATEGORICAL_FEATURES = PKL_PATH / "categorical_features.pkl"
METRICS_PATH = PKL_PATH / "metrics.json"

TEST_PATH = DATA_PATH / "heart_test.csv"
TRAIN_PATH = DATA_PATH / "heart_train.csv"

RESULT_PATH = RESULT_PATH / "predictions.csv"

# === Целевая переменная ===
TARGET = "Heart_Attack_Risk_(Binary)"

# === Список признаков с медицинским смыслом ===
MEDICAL_FEATURES = [
    'Cholesterol', 'Triglycerides', 'Systolic_blood_pressure', 'Diabetes',
    'Age', 'BMI', 'Troponin', 'CK_MB', 'Exercise_Activity', 'Blood_Pressure_Mean',
    'Family_History', 'Obesity', 'Previous_Heart_Problems', 'Smoking',
    'Alcohol_Consumption'
]
'''
Cholesterol: Высокий уровень холестерина — ключевой фактор риска
Triglycerides: Повышение триглицеридов связано с сердечно-сосудистыми заболеваниями
Systolic_blood_pressure: Артериальное давление — один из главных показателей риска
Blood_Pressure_Mean: Усреднённое значение давления — полезно для обобщения
Diabetes: Диабет увеличивает риск инфаркта в 2–4 раза
Exercise_Activity: Комбинация упражнений и активности — положительная корреляция
Age: Возраст — важнейший фактор риска
BMI: Чем выше ИМТ, тем больше нагрузка на сердце
Troponin: Биомаркер повреждения миокарда — напрямую указывает на патологию
CK-MB: Фермент, маркер повреждения сердца — тоже клинически значим
Family_History: Генетическая предрасположенность влияет на риск
Obesity: Ожирение — фактор риска сердечно-сосудистых заболеваний
Previous_Heart_Problems: Предыдущие проблемы → высокий риск повтора
Smoking: Курение — один из основных модифицируемых факторов риска
Alcohol_Consumption: Чрезмерное потребление алкоголя вредит сердцу
'''

# === Список признаков с индивидуальным смыслом ===
# Не подошли, поэтому убрал их из обучения
INDIRECT_FEATURES = [
    'Cholesterol_Triglycerides', 'Exercise_Hours_Per_Week', 'Physical_Activity_Days_Per_Week',
    'Stress_Level', 'Sleep_Hours_Per_Day', 'Diet', 'Gender',
    'Income',
]
'''
Cholesterol_Triglycerides: Произведение холестерина и триглицеридов — может усиливать эффект
Exercise_Hours_Per_Week: Кол-во часов тренировок — чем больше, тем меньше риск
Physical_Activity_Days_Per_Week: Дни активности — отрицательная корреляция, но логично
Stress_Level: Стресс влияет на здоровье сердца
Sleep_Hours_Per_Day: Недостаток сна — фактор риска
Diet: Тип диеты (например, средиземноморская vs западная)
Income: Социально-экономический статус влияет на доступ к лечению
Gender: Мужчины чаще болеют, чем женщины до менопаузы
'''
