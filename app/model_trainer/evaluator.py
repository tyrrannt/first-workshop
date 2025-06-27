from sklearn.metrics import f1_score, classification_report, confusion_matrix, roc_auc_score
import numpy as np

class ModelEvaluator:
    def __init__(self, model, X_val, y_val):
        self.model = model
        self.X_val = X_val
        self.y_val = y_val

    def find_best_threshold(self):
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
            # "best_params": best['best_params']
        }
