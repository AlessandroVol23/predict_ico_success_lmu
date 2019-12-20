import logging

import click
import numpy as np
import pandas as pd
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold

from src.models.catboost_model import CatBoostModel
from src.models.fit_model import FittingModel
from src.models.light_gbm import LightGbmModel
from src.models.naive_bayes import NaiveBayesModel
from src.models.utils import read_upsampling_feature_set, read_categorical_features, get_submission_number, \
    create_evaluation_file, write_results
from src.utils import read_feature_meta

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

training_models = [
    CatBoostModel,
    LightGbmModel,
    NaiveBayesModel
]


def train_final_classifier(X, y):
    hyperparam = {
        "bagging_temperature": 1.0,
        "border_count": 202,
        "depth": 5,
        "iterations": 705,
        "l2_leaf_reg": 30,
        "learning_rate": 0.29502864152955893,
        "random_strength": 10.0,
        "scale_pos_weight": 1.0,
        'logging_level': 'Verbose',
    }

    clf = CatBoostModel(hyperparam)
    clf.fit(X, y)
    return clf


def cross_validate_stacking(X_train, X_test, y_train):
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=123)
    oof_train = np.zeros(X_train.shape[0])
    oof_test = np.zeros(X_test.shape[0])
    oof_test_skf = np.empty((folds.n_splits, X_test.shape[0]))
    mcc_folds = []

    hyperparam = {
        'iterations': 2000,
        'eval_metric': 'MCC',
        'loss_function': 'Logloss',
        'use_best_model': True,
        'early_stopping_rounds': 300,
        'logging_level': 'Silent'
    }

    clf = CatBoostModel(hyperparam)
    logger.info("Start cross validating Stacking Model.")

    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(X_train, y_train)):
        trn_x, trn_y = X_train.iloc[trn_idx], y_train.iloc[trn_idx]
        val_x, val_y = X_train.iloc[val_idx], y_train.iloc[val_idx]

        clf.fit(trn_x, trn_y, val_x, val_y)

        probs = clf.predict_proba(val_x)
        if probs.shape[1] == 2:
            probs = probs[:, 1]
        oof_train[val_idx] = probs
        oof_train_abs = oof_train[val_idx].round()

        # Get MCC for validation set
        mcc = matthews_corrcoef(val_y, oof_train_abs)
        mcc_folds.append(mcc)

        logger.info('Fold %2d mcc : %.6f' %
                    (n_fold + 1, mcc))

        # Calculate submission predictions
        probs = clf.predict_proba(X_test)
        if probs.shape[1] == 2:
            probs = probs[:, 1]
        oof_test_skf[n_fold, :] = probs

    oof_test[:] = oof_test_skf.mean(axis=0)
    oof_test_abs = oof_test[:].round()
    mean_mcc = np.array(mcc_folds).mean()
    logger.info("Overall MCC was: {}".format(mean_mcc))
    return mean_mcc, oof_test_abs


def make_submission(fitting_model, next_submission_number):
    test_ids, sub_preds_abs = fitting_model.get_values()
    create_evaluation_file(test_ids, sub_preds_abs,
                           next_submission_number, True)


def make_result_file(model_name, feature_set_meta, feature_set_key, mean_mcc, next_submission_number, hp):
    # get name and params from underlying model

    write_results(feature_set_meta, feature_set_key, mean_mcc, model_name,
                  next_submission_number, hp)


@click.command()
@click.argument('feature_set_key')
def stack_models(feature_set_key):
    feature_set_meta = read_feature_meta()

    df_oof_train = pd.DataFrame()
    df_oof_test = pd.DataFrame()

    for current_model_class in training_models:
        current_model = current_model_class()

        if current_model.get_name() == 'catboost':
            hp = {
                'iterations': 2000,
                'eval_metric': 'MCC',
                'loss_function': 'Logloss',
                'use_best_model': True,
                'early_stopping_rounds': 300,
                'logging_level': 'Silent'
            }
            current_model.hyperparam = hp
        elif current_model.get_name() == 'lbm':
            hp = {
                'n_estimators': 2000,
                'learning_rate': '0.0035',
                'device': 'CPU'}
            current_model.hyperparam = hp

        upsampling = read_upsampling_feature_set(
            feature_set_meta, feature_set_key)

        categorical_features = read_categorical_features(
            feature_set_meta, feature_set_key)
        # Fitting model that trains and cross validates, takes the underlying model to train as a param
        fitting_model = FittingModel(
            feature_set_key, current_model, categorical_features, upsample=upsampling)

        # Get values from fitting model
        mean_mcc = fitting_model.cross_validation()
        oof_train, oof_test = fitting_model.get_oof()

        df_oof_train[current_model.get_name()] = oof_train
        df_oof_test[current_model.get_name()] = oof_test

    # Cross validate stacking model
    mean_mcc, oof_test_abs = cross_validate_stacking(df_oof_train, df_oof_test, fitting_model.y_train)

    next_submission_number = get_submission_number()

    make_submission(fitting_model, next_submission_number)

    model_names = []
    for m in training_models:
        cm = m()
        model_names.append(cm.get_name())

    make_result_file('stacking_' + '_'.join(model_names), feature_set_meta, feature_set_key, mean_mcc,
                     next_submission_number, hp)

    # final_cv = train_final_classifier(df_oof_train, fitting_model.y_train)
    logger.info("Stacked model trained!")


if __name__ == "__main__":
    stack_models()
