#! /bin/bash
# -*- coding: utf-8 -*-

import logging
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from lightgbm import LGBMClassifier
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import KFold
import numpy as np
from src.models.utils import read_feature_data
from sklearn.metrics import roc_auc_score

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


class LightGbmModel(object):

    def __init__(self):
        """Constructor for class LightGbmModel

        Parameters
        ----------
        X : DataFrame
            X_train Dataframe from build_features.py
        y : DataFrame
            y_train DataFrame from build_features.py
        """
        self.X_train, self.y_train, self.X_test = read_feature_data()

    def cross_validation(self):
        """Cross validation
        """
        # Modeling
        folds = KFold(n_splits=5, shuffle=True, random_state=123)
        oof_preds = np.zeros(self.X_train.shape[0])
        sub_preds = np.zeros(self.X_test.shape[0])
        for n_fold, (trn_idx, val_idx) in enumerate(folds.split(self.X_train)):
            trn_x, trn_y = self.X_train.iloc[trn_idx], self.y_train.iloc[trn_idx]
            val_x, val_y = self.X_train.iloc[val_idx], self.y_train.iloc[val_idx]

            clf = LGBMClassifier(
                n_estimators=2000,
                learning_rate=0.1,
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
                    eval_metric='auc', verbose=250, early_stopping_rounds=150
                    )

            oof_preds[val_idx] = clf.predict_proba(
                val_x, num_iteration=clf.best_iteration_)[:, 1]
            sub_preds += clf.predict_proba(test[features], num_iteration=clf.best_iteration_)[
                :, 1] / folds.n_splits

            print('Fold %2d AUC : %.6f' %
                  (n_fold + 1, roc_auc_score(val_y, oof_preds[val_idx])))
            del clf, trn_x, trn_y, val_x, val_y

    def train(self):
        """Train lightgbm model with some harcoded parameters
        """
        # Convert it into a format fitting for LGBM Multiclass
        train_set = lgb.Dataset(self.X_train, self.y_train)
        val_set = lgb.Dataset(self.X_test, self.y_test)

        parameters = {
            'application': 'binary',
            'objective': 'binary',
            'metric': 'auc',
            'is_unbalance': 'true',
            'boosting': 'gbdt',
            'num_leaves': 31,
            'feature_fraction': 0.5,
            'bagging_fraction': 0.5,
            'bagging_freq': 20,
            'learning_rate': 0.05,
            'verbose': 0
        }

        self.model = lgb.train(parameters,
                               train_set,
                               valid_sets=val_set,
                               num_boost_round=5000,
                               early_stopping_rounds=100)

    def evaluate(self):
        """Function to calculate MCC metric.
        """
        # Get prediction probabilities for test set
        preds = self.model.predict(
            self.X_test, num_iteration=self.model.best_iteration)

        # Probabilities to classes
        classes_preds = preds.round().astype(int)

        # Print mcc
        logger.info("MCC is; {}".format(
            matthews_corrcoef(self.y_test, classes_preds)))


def main():
    model = LightGbmModel()
    model.cross_validation()
    pass


if __name__ == "__main__":
    main()
