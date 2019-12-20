from sklearn.naive_bayes import GaussianNB
import numpy as np

from src.models.base_model import BaseModel


class NaiveBayesModel(BaseModel):

    def __init__(self):
        self.hyperparameter={}
        pass

    def get_name(self):
        return 'naive_bayes'

    def get_params(self):
        return self.hyperparameter

    def get_model(self, reinitialize=True):
        if (reinitialize):
            self.model = GaussianNB()
        return self.model

    def fit(self, trn_x, trn_y, val_x=None, val_y=None):
        self.model.fit(trn_x, trn_y)

    def predict_proba(self, test_set):
        preds = self.model.predict_proba(test_set)
        return preds
