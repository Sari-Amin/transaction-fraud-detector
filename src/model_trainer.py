from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier

class FraudModelTrainer:
    def __init__(self, model_name):
        if model_name == "logistic":
            self.model = LogisticRegression(max_iter=1000)
        elif model_name == "xgboost":
            self.model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
        elif model_name == "random_forest":
            self.model = RandomForestClassifier()
        else:
            raise ValueError("Unsupported model type")
        self.model_name = model_name

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def predict_proba(self, X_test):
        return self.model.predict_proba(X_test)[:, 1]

    def get_model(self):
        return self.model
