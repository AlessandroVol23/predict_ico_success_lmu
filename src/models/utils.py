#! /bin/bash
# -*- coding: utf-8 -*-

import logging
import datetime

import numpy as np
import pandas as pd

from src.utils import read_feature_meta

logger = logging.getLogger(__name__)


def read_feature_data(feature_set="1"):
    x_train = 'data/features/feature_sets/features_x_train_' + feature_set + '.csv'
    y_train = 'data/features/feature_sets/features_y_train_' + feature_set + '.np'
    x_test = 'data/features/feature_sets/features_x_test_' + feature_set + '.csv'

    x_train = pd.read_csv(x_train)
    # y_train = pd.DataFrame(np.fromfile(y_train))
    y_train = pd.DataFrame(np.fromfile(y_train, dtype=int))
    y_train.columns = ['success']
    logger.info("Unique of y_train: {}".format(y_train.success.value_counts()))
    x_test = pd.read_csv(x_test)

    return x_train, y_train, x_test


def upsample_data(x, y, upsampling):
    df = x.assign(success=y)
    df_success = df.loc[df.success == 1]
    quantity = int(len(df_success) * upsampling)
    to_append = df_success.sample(quantity, random_state=123, replace=True)
    df_upsampled = df.append(to_append)
    assert len(df_upsampled) == (len(df) + len(to_append)
                                 ), "Length is wrong after upsampling."

    trn_X = df_upsampled.drop('success', axis=1)
    trn_y = df_upsampled.success
    return trn_X, trn_y


def read_upsampling_feature_set(feature_set_meta, feature_set_key):
    feature_set = feature_set_meta[feature_set_key]
    for feature in feature_set:
        if 'meta' in feature:
            try:
                upsampling = float(feature['meta']['upsampling'])
                return upsampling
            except ValueError:
                logger.warning(
                    "Won't upsample because no float value was provided!")


def read_categorical_features(feature_set_meta, feature_set_key):
    feature_set = feature_set_meta[feature_set_key]
    categorical_features = []
    for feature in feature_set:
        if 'meta' in feature:
            continue
        feature_name = feature["column"]
        feature_type = feature["type"]
        if feature_type == "categorical":
            categorical_features.append(feature_name)

    return categorical_features


def read_result_csv():
    try:
        df = pd.read_csv('data/results/result.csv')
    except FileNotFoundError:
        logger.info("File wasn't found. Create new one")
        df = pd.DataFrame()
    return df


def write_result_csv(result, result_ser):
    df = result.append(result_ser, ignore_index=True)
    df.to_csv('data/results/result.csv', index=None)


def get_submission_number():
    with open("SUBMISSION_NUMBER", "r") as f:
        return f.readline()


def increment_submission_number(current_number=0):
    new_build_number = int(current_number) + 1
    with open("SUBMISSION_NUMBER", "w") as f:
        f.write(str(new_build_number))


def create_evaluation_file(test_ids, sub_preds_abs, next_submission_number, increment=True):
    if increment:
        increment_submission_number(next_submission_number)

    df_submission = pd.DataFrame(
        [test_ids.values, sub_preds_abs]).transpose()
    df_submission.columns = ['OBS_ID', 'success']
    df_submission['OBS_ID'] = df_submission.OBS_ID.astype(int)
    df_submission['success'] = df_submission.success.astype(int)
    fileName = 'data/submissions/submission' + next_submission_number + '.csv'
    df_submission.to_csv(fileName, index=None)
    logger.info("Write submission file to: {}".format(fileName))

def write_results(feature_set_meta, feature_set_number, mean_mcc, model_name, next_submission_number,
                   hyperparam):
    feature_sets = read_feature_meta()
    if (feature_set_number in feature_sets):
        feature_set = feature_sets[feature_set_number]
    else:
        feature_sets = read_feature_meta(True)
        feature_set = feature_sets[feature_set_number]

    upsampling = read_upsampling_feature_set(
        feature_set_meta, feature_set_number)

    result = read_result_csv()

    if hyperparam is '':
        hyperparam={}

    result_ser = pd.Series({
        'timestamp': datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
        'feature_set': feature_set_number,
        'features': feature_set,
        'model_name': model_name,
        'submission_number': next_submission_number,
        'mcc_cv': mean_mcc,
        'upsampling': upsampling,
        'submission_score': 'TO_FILL',
        'hp_iterations': hyperparam.get('iterations', 'NA'),
        'hp_early_stopping_rounds': hyperparam.get('early_stopping_rounds', 'NA'),
        'hp_n_estimators': hyperparam.get('n_estimators', 'NA'),
        'hp_learning_rate': hyperparam.get('learning_rate', 'NA'),
        'hp_loss_function': hyperparam.get('loss_function', 'NA'),
        'hp_use_best_model': hyperparam.get('use_best_model', 'NA')
    })
    write_result_csv(result, result_ser)
