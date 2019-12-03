#! /bin/bash
# -*- coding: utf-8 -*-

import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from src.models.utils import upsample_data

from src.models.utils import read_feature_data



logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


class FittingModel(object):

    def __init__(self, feature_set,model, upsample=None):
        """Constructor for class LightGbmModel

        Parameters
        ----------
        X : DataFrame
            X_train Dataframe from build_features.py
        y : DataFrame
            y_train DataFrame from build_features.py
        """
        self.X_train, self.y_train, self.X_test = read_feature_data(
            feature_set)
        self.test_ids = self.X_test['OBS_ID']
        self.X_test = self.X_test.drop('OBS_ID', axis=1)
        self.upsample = upsample
        self.model = model
        self.traning_model = self.model.get_model()
        logger.info("X_train shape: {}".format(self.X_train.shape))
        logger.info("y_train shape: {}".format(self.y_train.shape))
        logger.info("x_test shape: {}".format(self.X_test.shape))

    def get_values(self):

        return self.test_ids, self.sub_preds_abs

    def cross_validation(self):
        """Cross validation
        """
        # Modeling
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

            self.model.fit(trn_x, trn_y,val_x,val_y)
            oof_pred_abs = self.model.predict_proba(oof_preds,sub_preds,self.X_test,folds,val_idx,val_x )
            unique_elements, counts_elements = np.unique(
                oof_pred_abs, return_counts=True)
            logger.debug("unique elements: {}: counts_elements: {}".format(
                unique_elements, counts_elements))
            mcc = matthews_corrcoef(val_y, oof_pred_abs[val_idx])
            logger.info('Fold %2d mcc : %.6f' %
                (n_fold + 1, mcc))

            mcc_folds.append(mcc)
            del trn_x, trn_y, val_x, val_y

        self.sub_preds_abs = sub_preds.round()
        mean_mcc = np.array(mcc_folds).mean()
        print("Overall MCC was: {}".format(mean_mcc))
        return mean_mcc