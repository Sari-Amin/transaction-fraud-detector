from sklearn.metrics import f1_score, precision_recall_curve, average_precision_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import seaborn as sns

class ModelEvaluator:
    def __init__(self, y_true, y_pred, y_proba, model_name="Model"):
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_proba = y_proba
        self.model_name = model_name

    def print_metrics(self):
        f1 = f1_score(self.y_true, self.y_pred)
        auc_pr = average_precision_score(self.y_true, self.y_proba)
        auc_roc = roc_auc_score(self.y_true, self.y_proba)

        print(f"\n📊 Evaluation Metrics for {self.model_name}:")
        print(f"F1 Score: {f1:.4f}")
        print(f"AUC-PR: {auc_pr:.4f}")
        print(f"AUC-ROC: {auc_roc:.4f}")

    def plot_confusion_matrix(self):
        cm = confusion_matrix(self.y_true, self.y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm)
        disp.plot(cmap='Blues')
        plt.title(f"{self.model_name} - Confusion Matrix")
        plt.show()

    def plot_precision_recall(self):
        precision, recall, _ = precision_recall_curve(self.y_true, self.y_proba)
        plt.plot(recall, precision, label=f'{self.model_name} (AUC={average_precision_score(self.y_true, self.y_proba):.2f})')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title(f"{self.model_name} - Precision Recall Curve")
        plt.legend()
        plt.grid()
        plt.show()

    def plot_roc_curve(self):
        fpr, tpr, _ = roc_curve(self.y_true, self.y_proba)
        plt.plot(fpr, tpr, label=f'{self.model_name} (AUC={roc_auc_score(self.y_true, self.y_proba):.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title(f"{self.model_name} - ROC Curve")
        plt.legend()
        plt.grid()
        plt.show()
