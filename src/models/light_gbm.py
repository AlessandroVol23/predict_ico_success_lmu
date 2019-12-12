#! /bin/bash
# -*- coding: utf-8 -*-

import logging
import os

from src.models.base_model import BaseModel
import lightgbm as lgb
from lightgbm import LGBMClassifier

class LightGbmModel(BaseModel):

    def __init__(self):
        self. hyperparam = {
            'n_estimators': 2000,
            'learning_rate': '0.003',
            'device':'GPU'
        }
        self.model = LGBMClassifier(
                **self.hyperparam
        )


    def get_name(self):
        return "lbm"
    
    def get_params(self):
        return self.hyperparam

    def get_model(self, reinitialize = False):
       
        if (reinitialize):
            self.model = LGBMClassifier(
                **self.hyperparam
            )
        return self.model
    def fit(self, trn_x, trn_y,val_x, val_y ):
        self.model.fit(trn_x, trn_y,
                        eval_set=[(trn_x, trn_y), (val_x, val_y)],
                        eval_metric='binary_logloss', verbose=250, early_stopping_rounds=100
                        )
    def predict_proba(self,oof_preds,sub_preds,X_test,folds,val_idx,val_x ):
        oof_preds[val_idx] = self.model.predict_proba(
            val_x, num_iteration=self.model.best_iteration_)[:, 1]
        sub_preds += self.model.predict_proba(X_test, num_iteration=self.model.best_iteration_)[
            :, 1] / folds.n_splits

        oof_pred_abs = oof_preds.round()

        return oof_pred_abs
