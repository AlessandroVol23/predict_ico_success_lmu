#! /bin/bash
# -*- coding: utf-8 -*-

import logging
import os

from src.models.base_model import BaseModel
from catboost import CatBoostClassifier, Pool


class CatBoostModel(BaseModel):

    def __init__(self):
        self. hyperparam = {
            'iterations':500,
             'depth':2,
             'eval_metric':'MCC',
             'loss_function':'Logloss',
             'verbose':True
        }
        self.model =CatBoostClassifier(
            **self.hyperparam
        )


    def get_name(self):
        return "catboost"
    
    def get_params(self):
        return self.hyperparam

    def get_model(self, reinitialize = False):
       
        if (reinitialize):
            self.model =CatBoostClassifier(
            **self.hyperparam
        )
        return self.model
    def fit(self, trn_x, trn_y,val_x, val_y ):
        self.model.fit(trn_x, trn_y,
                        eval_set=[(trn_x, trn_y), (val_x, val_y)],
                        )
    def predict_proba(self,oof_preds,sub_preds,X_test,folds,val_idx,val_x ):
        oof_preds[val_idx] = self.model.predict_proba(
            val_x)[:, 1]
        sub_preds += self.model.predict_proba(X_test)[
            :, 1] / folds.n_splits

        oof_pred_abs = oof_preds.round()

        return oof_pred_abs
