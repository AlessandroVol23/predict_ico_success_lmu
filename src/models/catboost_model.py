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


"""

    def cross_validation(self):

        # Modeling
        folds = KFold(n_splits=5, shuffle=True, random_state=123)
        # StratifiedKFold
        # folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
        # skf.get_n_splits(self.X_train, self.y_train)
    
        oof_preds = np.zeros(self.X_train.shape[0])
        sub_preds = np.zeros(self.X_test.shape[0])
        mcc_folds = []
        for n_fold, (trn_idx, val_idx) in enumerate(folds.split(self.X_train, self.y_train)):
            trn_x, trn_y = self.X_train.iloc[trn_idx], self.y_train.iloc[trn_idx]
            if self.upsample:
                trn_x, trn_y = upsample_data(trn_x, trn_y, self.upsample)

            val_x, val_y = self.X_train.iloc[val_idx], self.y_train.iloc[val_idx]


            clf.fit(trn_x, trn_y,
                    eval_set=[(trn_x, trn_y), (val_x, val_y)],
                    eval_metric='binary_logloss', verbose=250, early_stopping_rounds=300
                    )

            oof_preds[val_idx] = clf.predict_proba(
                val_x, num_iteration=clf.best_iteration_)[:, 1]
            sub_preds += clf.predict_proba(self.X_test, num_iteration=clf.best_iteration_)[
                :, 1] / folds.n_splits

            oof_pred_abs = oof_preds.round()

            unique_elements, counts_elements = np.unique(
                oof_pred_abs, return_counts=True)
            logger.info("unique elements: {}: counts_elements: {}".format(
                unique_elements, counts_elements))
            mcc = matthews_corrcoef(val_y, oof_pred_abs[val_idx])
            print('Fold %2d mcc : %.6f' %
                  (n_fold + 1, mcc))

            mcc_folds.append(mcc)
            del clf, trn_x, trn_y, val_x, val_y

        self.sub_preds_abs = sub_preds.round()
        mean_mcc = np.array(mcc_folds).mean()
        print("Overall MCC was: {}".format(mean_mcc))
        return mean_mcc, hyperparam

"""
