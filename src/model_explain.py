import shap

class SHAPInterpreter:
    def __init__(self, model, X_sample, feature_names):
        self.explainer = shap.Explainer(model, X_sample)
        self.shap_values = self.explainer(X_sample)
        self.X_sample = X_sample
        self.feature_names = feature_names

    def summary(self):
        shap.summary_plot(self.shap_values, self.X_sample, feature_names=self.feature_names)

    def force(self, i=0):
        shap.initjs()
        shap.plots.force(self.shap_values[i])  

    def waterfall(self, i=0):
        shap.plots.waterfall(self.shap_values[i])
