#! /bin/bash
# -*- coding: utf-8 -*-

import logging
import os
from src.models.base_model import BaseModel
from catboost import CatBoostClassifier, Pool


class CatBoostModel(BaseModel):

    def __init__(self, hyperparams=None):

        if hyperparams is None:
            self.hyperparam = {
                'iterations': 2000,
                # 'depth': 8,
                'eval_metric': 'MCC',
                'loss_function': 'Logloss',
                # 'loss_function': 'CrossEntropy',
                #  'learning_rate': 0.1,
                'use_best_model': True,
                'early_stopping_rounds': 300,
                'logging_level': 'Verbose'
            }
            self.model = CatBoostClassifier(
                **self.hyperparam
            )
        else:
            self.hyperparam = hyperparams

        self.model = self.get_model(reinitialize=True)

    def get_name(self):
        return "catboost"

    def get_params(self):
        return self.hyperparam

    def get_model(self, reinitialize=False):
        if (reinitialize):
            self.model = CatBoostClassifier(
                **self.hyperparam
            )
        return self.model

    def fit(self, trn_x, trn_y, val_x=None, val_y=None, categorical_features=[]):
        if val_x is not None:
            self.model.fit(trn_x, trn_y,
                           eval_set=[(val_x, val_y)],
                           cat_features=categorical_features,
                           )
        else:
            self.model.fit(trn_x, trn_y)

    def predict_proba(self, test_set):
        preds = self.model.predict_proba(test_set)
        return preds

    def get_feature_importance(self, X_test, method="ShapValues", preds_test=[], categorical_features_indices=[]):

        if method == "ShapValues":
            return self.model.get_feature_importance(Pool(X_test, cat_features=categorical_features_indices),
                                                     type="ShapValues")
        else:
            print("wrong here " + method)
            return self.model.get_feature_importance(Pool(X_test,  label=preds_test, cat_features=categorical_features_indices),
                                                     type=method)
