#! /bin/bash
# -*- coding: utf-8 -*-

from lightgbm import LGBMClassifier

from src.models.base_model import BaseModel


class LightGbmDartModel(BaseModel):

    def __init__(self):
        self.hyperparam = {
            'n_estimators': 2000,
            'learning_rate': '0.1',
            'device': 'CPU',
            'boosting': 'dart'
        }
        self.model = LGBMClassifier(
            **self.hyperparam
        )

    def get_name(self):
        return "lbm_dart"

    def get_params(self):
        return self.hyperparam

    def get_model(self, reinitialize=True):

        if (reinitialize):
            self.model = LGBMClassifier(
                **self.hyperparam
            )
        return self.model

    def fit(self, trn_x, trn_y, val_x=None, val_y=None, categorical_features=[]):

        if val_x is not None:
            self.model.fit(trn_x, trn_y,
                           eval_set=[(trn_x, trn_y), (val_x, val_y)],
                           eval_metric='binary_logloss')
        else:
            self.model.fit(trn_x, trn_y)

    def predict_proba(self, test_set):
        preds = self.model.predict_proba(test_set)
        return preds
