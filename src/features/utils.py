#! /bin/bash
# -*- coding: utf-8 -*-

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def read_processed_data(path_bitcoin_df='data/processed/df_bitcoin_pp.csv',
                        path_training_df='data/processed/df_train_pp.csv',
                        path_test_df='data/processed/df_test_pp.csv'):
    """Function to read in data

    Parameters
    ----------
    path_bitcoin_df : str, optional
        Path to bitcoin set, by default 'data/raw/1_training_data_sets/1_bitcoin_price_data_set.csv'
    path_training_df : str, optional
        Path to training set, by default 'data/raw/1_training_data_sets/1_training_data.csv'
    path_test_df : str, optional
        Path to training set, by default 'data/raw/2_classification_data.csv'


    Returns
    -------
    tuple (df, df, df)
        df_bitcoin, df, df_test
    """
    df_bitcoin = pd.read_csv(
        path_bitcoin_df, encoding="ISO-8859-1", delimiter=';')
    df = pd.read_csv(path_training_df, encoding="ISO-8859-1")

    df_test = pd.read_csv(path_test_df, encoding="ISO-8859-1")

    logger.info("Shape of df_bitcoin: {}".format(df_bitcoin.shape))
    logger.info("Shape of df: {}".format(df.shape))
    return df_bitcoin, df, df_test


def save_x_y(X_train, y_train, X_test):
    logger.info("Saving X_train, y_train and")
    X_train.tofile('data/processed/X_train.np')
    y_train.tofile('data/processed/y_train.np')
    X_test.tofile('data/processed/X_test.np')
