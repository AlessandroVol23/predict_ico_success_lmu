from sklearn.linear_model import LogisticRegression

from src.models.base_model import BaseModel


class LogisticRegressionModel(BaseModel):

    def __init__(self):
        self.hyperparameter = {
            "max_iter": 10000
        }
        pass

    def get_name(self):
        return 'logistic_regression'

    def get_params(self):
        return self.hyperparameter

    def get_model(self, reinitialize=True):
        if (reinitialize):
            self.model = LogisticRegression(**self.hyperparameter)
        return self.model

    def fit(self, trn_x, trn_y, val_x=None, val_y=None):
        self.model.fit(trn_x, trn_y)

    def predict_proba(self, test_set):
        preds = self.model.predict_proba(test_set)
        return preds
