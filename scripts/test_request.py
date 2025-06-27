import requests

from app.config import TEST_PATH

url = "http://127.0.0.1:8000/predict_file"
try:
    with open(TEST_PATH, "rb") as f:
        files = {"file": ("test.csv", f)}
        response = requests.post(url, files=files)
except FileNotFoundError:
    print("Ошибка: файл не найден.")
    exit(1)

if response.status_code == 200:
    print("Ответ от сервера:")
    print(response.json())
else:
    print("Ошибка:", response.status_code)
    print(response.text)
