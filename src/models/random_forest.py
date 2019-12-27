from sklearn.ensemble import RandomForestClassifier

from src.models.base_model import BaseModel


class RandomForestModel(BaseModel):

    def __init__(self, hyperparams = None):
        if hyperparams == None:

            self.hyperparameter = {
                "n_estimators": 500
            }
        else:
            self.hyperparameter={
                **hyperparams
            }
       #  self.model = RandomForestClassifier()


    def get_name(self):
        return 'random_forest'

    def get_params(self):
        return self.model.get_params()

    def get_model(self, reinitialize=True):
        if reinitialize:
            self.model = RandomForestClassifier(**self.hyperparameter)
        return self.model

    def fit(self, trn_x, trn_y, val_x=None, val_y=None, categorical_features=[]):
        self.model.fit(trn_x, trn_y)

    def predict_proba(self, test_set):
        return self.model.predict_proba(test_set)
