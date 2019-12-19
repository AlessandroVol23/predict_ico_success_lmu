#! /bin/bash
# -*- coding: utf-8 -*-

import logging
import os

import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split
from src.models.utils import upsample_data
from time import time
from joblib import dump
import shap
import matplotlib.pyplot as plt

from src.models.utils import read_feature_data
from src.utils import create_folder

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


class FittingModel(object):

    def __init__(self, feature_set, model, categorical_features, upsample=None):
        """Constructor for class LightGbmModel

        Parameters
        ----------
        X : DataFrame
            X_train Dataframe from build_features.py
        y : DataFrame
            y_train DataFrame from build_features.py
        """
        self.base_time_stamp = str(int(time()))
        self.X_train, self.y_train, self.X_test = read_feature_data(
            feature_set)
        self.feature_set = feature_set
        self.test_ids = self.X_test['OBS_ID']
        self.X_test = self.X_test.drop('OBS_ID', axis=1)
        self.upsample = upsample
        # self.categorical_features = np.array(list(map(lambda x: self.X_train.columns.get_loc(x), categorical_features)))
        self.model = model
        self.traning_model = self.model.get_model(reinitialize=True)
        logger.info("X_train shape: {}".format(self.X_train.shape))
        logger.info("y_train shape: {}".format(self.y_train.shape))
        logger.info("x_test shape: {}".format(self.X_test.shape))

    def get_values(self):
        return self.test_ids, self.sub_preds_abs

    def train_final_model(self):
        self.model.fit(self.X_train, self.y_train)
        logger.info("Model trained!")

    def predict_test_set(self):
        preds_test = self.model.predict_proba(self.X_test)
        return preds_test

    def _get_model_file_name(self):
        return 'models/local/' + self.model.get_name() + '_' + self.feature_set + self.base_time_stamp + '.sav'

    def _get_figure_file_name(self, method, path):
        return os.path.join(path, ('feature_importance' + '_' + method + '.png'))

    def get_oof(self):
        return self.oof_train, self.oof_test

    def save_current_model(self):
        filename = self._get_model_file_name()
        dump(self.model.get_model(), filename)
        logger.info("Saved model under {}".format(filename))
        return filename

    def save_feature_importance(self, method, preds_test=[]):
        logger.info(method)
        path = os.path.join('reports/figures/local/', (self.model.get_name() + '_' + self.feature_set))
        create_folder(path)
        filename = self._get_figure_file_name(method, path)
        explainer = shap.TreeExplainer(self.model.get_model(reinitialize=False))
        shap_values = explainer.shap_values(self.X_train)

        if method == "shap":
            # shap_values = shap_values[:, :-1]
            shap.summary_plot(shap_values, self.X_train, show=False)
            plt.savefig(self._get_figure_file_name('shap', path), bbox_inches='tight')

        elif method == 'summary':
            shap.summary_plot(shap_values, self.X_train, plot_type='bar', show=False)
            plt.savefig(self._get_figure_file_name('summary', path), bbox_inches='tight')

        elif method == 'feature_importance':
            feature_score = pd.DataFrame(list(zip(self.X_test.dtypes.index, shap_values.mean(axis=0))),
                                         columns=['Feature', 'Score'])
            feature_score = feature_score.sort_values(by='Score', ascending=False, inplace=False, kind='quicksort',
                                                      na_position='last')
            plt.rcParams["figure.figsize"] = (12, 7)
            ax = feature_score.plot('Feature', 'Score', kind='bar', color='c')
            ax.set_title("Feature Importance using {}".format(method), fontsize=14)
            ax.set_xlabel("features")
            plt.savefig(self._get_figure_file_name(method, path), bbox_inches='tight')

    def cross_validation(self):
        """Cross validation
        """
        # Modeling
        # folds = KFold(n_splits=5, shuffle=True, random_state=123)
        # StratifiedKFold
        folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
        # skf.get_n_splits(self.X_train, self.y_train)

        self.oof_train = np.zeros(self.X_train.shape[0])
        self.oof_test = np.zeros(self.X_test.shape[0])
        oof_test_skf = np.empty((folds.n_splits, self.X_test.shape[0]))
        mcc_folds = []
        for n_fold, (trn_idx, val_idx) in enumerate(folds.split(self.X_train, self.y_train)):

            trn_x, trn_y = self.X_train.iloc[trn_idx], self.y_train.iloc[trn_idx]

            if self.upsample:
                trn_x, trn_y = upsample_data(trn_x, trn_y, self.upsample)

            val_x, val_y = self.X_train.iloc[val_idx], self.y_train.iloc[val_idx]

            self.model.fit(trn_x, trn_y, val_x, val_y)

            # Get probabilities for validation set
            # Just get probability for first class
            probs = self.model.predict_proba(val_x)
            if probs.shape[1] == 2:
                probs = probs[:, 1]
            self.oof_train[val_idx] = probs
            self.oof_train_abs = self.oof_train[val_idx].round()

            # Get MCC for validation set
            mcc = matthews_corrcoef(val_y, self.oof_train_abs)

            # Calculate submission predictions
            probs = self.model.predict_proba(self.X_test)
            if probs.shape[1] == 2:
                probs = probs[:, 1]
            oof_test_skf[n_fold, :] = probs

            logger.info('Fold %2d mcc : %.6f' %
                        (n_fold + 1, mcc))

            mcc_folds.append(mcc)
            del trn_x, trn_y, val_x, val_y

        self.oof_test[:] = oof_test_skf.mean(axis=0)
        self.oof_test_abs = self.oof_test[:].round()
        mean_mcc = np.array(mcc_folds).mean()
        print("Overall MCC was: {}".format(mean_mcc))
        return mean_mcc
