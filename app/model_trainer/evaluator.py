from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
import numpy as np


class ModelEvaluator:
    def __init__(self, model, X_val, y_val):
        """
        Инициализирует объект оценки модели.

        Parameters
        ----------
        model : object
            Обученная модель, поддерживающая метод `predict_proba`.
        X_val : array-like
            Валидационные признаки.
        y_val : array-like
            Валидационные метки классов.
        """
        self.model = model
        self.X_val = X_val
        self.y_val = y_val

    def find_best_threshold(self):
        """
        Определяет оптимальный порог для бинарной классификации, максимизирующий F1-меру.

        Эта функция вычисляет вероятности принадлежности к положительному классу (класс 1)
        с помощью метода `predict_proba` обученной модели. Затем перебираются различные пороговые значения,
        и для каждого из них рассчитывается F1-мера на валидационной выборке. Порог с максимальным значением
        F1-меры выбирается как лучший.

        Returns
        -------
        float
            Лучший порог (threshold) для бинарной классификации, который обеспечивает максимальное значение F1-меры.
        """
        probs = self.model.predict_proba(self.X_val)[:, 1]
        best_threshold, best_f1 = 0.5, 0
        for t in np.arange(0.1, 0.9, 0.01):
            preds = (probs > t).astype(int)
            score = f1_score(self.y_val, preds)
            if score > best_f1:
                best_f1 = score
                best_threshold = t
        return best_threshold

    def evaluate(self, threshold):
        """
        Оценивает качество модели на валидационной выборке при заданном пороге.

        Функция рассчитывает вероятности положительного класса с помощью `predict_proba`,
        затем преобразует их в бинарные предсказания на основе переданного порога.
        Далее вычисляются метрики: F1-мера, точность, полнота, ROC AUC и матрица ошибок.

        Parameters
        ----------
        threshold : float
            Пороговое значение для бинарной классификации. Значения вероятностей, превышающие
            этот порог, считаются положительным классом (1), остальные — отрицательным (0).

        Returns
        -------
        dict
            Словарь с ключами:
            - best_threshold : float
                Использованный порог для классификации.
            - f1_score : float
                F1-мера для положительного класса.
            - precision : float
                Точность для положительного класса.
            - recall : float
                Полнота для положительного класса.
            - roc_auc : float
                Значение площади под ROC-кривой.
            - confusion_matrix : list of lists
                Матрица ошибок в виде списка списков.
            - report : dict
                Полный отчет о классификации в формате словаря.
        """
        probs = self.model.predict_proba(self.X_val)[:, 1]
        preds = (probs > threshold).astype(int)
        report = classification_report(self.y_val, preds, output_dict=True, zero_division=0)
        cm = confusion_matrix(self.y_val, preds)
        auc = roc_auc_score(self.y_val, probs)
        return {
            "best_threshold": float(threshold),
            "f1_score": report["1"]["f1-score"],
            "precision": report["1"]["precision"],
            "recall": report["1"]["recall"],
            "roc_auc": auc,
            "confusion_matrix": cm.tolist(),
            "report": report,
        }
