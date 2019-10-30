#! /bin/bash
# -*- coding: utf-8 -*-

import logging
from sklearn.model_selection import train_test_split
import lightgbm as lgb
from sklearn.metrics import matthews_corrcoef

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


class LightGbmModel(object):

    def __init__(self, X, y):
        """Constructor for class LightGbmModel

        Parameters
        ----------
        X : DataFrame
            X_train Dataframe from build_features.py
        y : DataFrame
            y_train DataFrame from build_features.py
        """
        self.X = X
        self.y = y

    def preprocess(self):
        """Preprocess data for model.
        At the moment just creating a test / validation split.
        """
        # Create train / test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42)

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
