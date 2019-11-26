#! /bin/bash
# -*- coding: utf-8 -*-

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def read_feature_data(feature_set="1"):
    x_train = 'data/features/features_x_train_'+feature_set+'.csv'
    y_train = 'data/features/features_y_train_'+feature_set+'.np'
    x_test = 'data/features/features_x_test_'+feature_set+'.csv'
    print(x_train)
    x_train = pd.read_csv(x_train)
    #y_train = pd.DataFrame(np.fromfile(y_train))
    y_train = pd.DataFrame(np.fromfile(y_train, dtype=int))
    x_test = pd.read_csv(x_test)
    #print("UNIQUE: {}".format(np.unique(y_train, return_counts=True)))
    print("UNIQUE: {}".format(y_train.groupby(0).count()))
    return x_train, y_train, x_test


def upsample_data(x, y, upsampling):
    df = x.assign(success=y)
    df_success = df.loc[df.success == 1]
    quantity = int(len(df_success) * upsampling)
    to_append = df_success.sample(quantity, random_state=123)
    df_upsampled = df.append(to_append)
    assert len(df_upsampled) == (len(df) + len(to_append)
                                 ), "Length is wrong after upsampling."

    trn_X = df.drop('success', axis=1)
    trn_y = df.success
    return trn_X, trn_y


def create_submission_csv(model, X_test, path_to_save_csv):
    """Function to create submission csv to upload on Kaggle.

    Parameters
    ----------
    model : model
        Pre-trained model
    X_test : DataFrame
        X_test DataFrame withou labels. Not the validation set!
    path_to_save_csv : str
        String where to save csv file.
    """
    # Get prediction probabilities for test set
    preds = model.predict(
        X_test, num_iteration=model.best_iteration)

    # Probabilities to classes
    classes_preds = preds.round().astype(int)
    df_submission = pd.DataFrame(classes_preds)
