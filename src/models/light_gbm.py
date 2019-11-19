#! /bin/bash
# -*- coding: utf-8 -*-

import logging
import os

import click
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef, roc_auc_score
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split

import lightgbm as lgb
from lightgbm import LGBMClassifier
from src.models.utils import read_feature_data

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


class LightGbmModel(object):

    def __init__(self,feature_set):
        """Constructor for class LightGbmModel

        Parameters
        ----------
        X : DataFrame
            X_train Dataframe from build_features.py
        y : DataFrame
            y_train DataFrame from build_features.py
        """
        self.X_train, self.y_train, self.X_test = read_feature_data(feature_set)
        self.test_ids = self.X_test['OBS_ID']
        self.X_test = self.X_test.drop('OBS_ID', axis=1)

        logger.info("X_train shape: {}".format(self.X_train.shape))
        logger.info("y_train shape: {}".format(self.y_train.shape))
        logger.info("x_test shape: {}".format(self.X_test.shape))

    def cross_validation(self):
        """Cross validation
        """
        # Modeling
        # folds = KFold(n_splits=5, shuffle=True, random_state=123)
        # StratifiedKFold
        folds = StratifiedKFold(n_splits=10, shuffle=True, random_state=123)
        # skf.get_n_splits(self.X_train, self.y_train)

        oof_preds = np.zeros(self.X_train.shape[0])
        sub_preds = np.zeros(self.X_test.shape[0])
        mcc_folds = []
        for n_fold, (trn_idx, val_idx) in enumerate(folds.split(self.X_train, self.y_train)):
            trn_x, trn_y = self.X_train.iloc[trn_idx], self.y_train.iloc[trn_idx]
            val_x, val_y = self.X_train.iloc[val_idx], self.y_train.iloc[val_idx]

            clf = LGBMClassifier(
                n_estimators=2000,
                learning_rate=0.01,
                num_leaves=123,
                colsample_bytree=.8,
                subsample=.9,
                max_depth=15,
                reg_alpha=.1,
                reg_lambda=.1,
                min_split_gain=.01,
                min_child_weight=2
            )

            clf.fit(trn_x, trn_y,
                    eval_set=[(trn_x, trn_y), (val_x, val_y)],
                    eval_metric='binary_logloss', verbose=250, early_stopping_rounds=150
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
        print("Overall MCC was: {}".format(np.array(mcc_folds).mean()))

    def get_submission_number(self):
        with open("SUBMISSION_NUMBER", "r") as f:
            return f.readline()

    def increment_submission_number(self, current_number=0):
        new_build_number = int(current_number) + 1
        print('New submission number is now: %2d' % (new_build_number))
        with open("SUBMISSION_NUMBER", "w") as f:
            f.write(str(new_build_number))

    # TODO get current submission_number and increment it after writing
    def create_evaluation_file(self, increment=True):
        next_submission_number = self.get_submission_number()
        if increment:
            self.increment_submission_number(next_submission_number)

        df_submission = pd.DataFrame(
            [self.test_ids.values, self.sub_preds_abs]).transpose()
        df_submission.columns = ['OBS_ID', 'success']
        df_submission['OBS_ID'] = df_submission.OBS_ID.astype(int)
        df_submission['success'] = df_submission.success.astype(int)
        fileName = 'data/submissions/submission' + next_submission_number + '.csv'
        df_submission.to_csv(fileName, index=None)

@click.command()
@click.argument('feature_set')
def main(feature_set):
    model = LightGbmModel(feature_set)
    model.cross_validation()
    model.create_evaluation_file()


if __name__ == "__main__":
    main()
