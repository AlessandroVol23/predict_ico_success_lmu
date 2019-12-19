#! /bin/bash
# -*- coding: utf-8 -*-

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def read_processed_data(path_bitcoin_df='data/processed/df_bitcoin_pp.csv',
                        path_training_df='data/processed/df_train_pp.csv',
                        path_test_df='data/processed/df_test_pp.csv',
                        path_gemini_btc_usd='data/processed/df_gem_btc_usd.csv',
                        path_gemini_eth_usd='data/processed/df_gem_eth_usd.csv',
                        path_gemini_ltc_usd='data/processed/df_gem_ltc_usd.csv',
                        path_icobench='data/processed/df_icobench.csv'):
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
    df_bitcoin = pd.read_csv(path_bitcoin_df)
    df = pd.read_csv(path_training_df, encoding="ISO-8859-1")

    df_test = pd.read_csv(path_test_df, encoding="ISO-8859-1")
    df_gem_btc_usd = pd.read_csv(path_gemini_btc_usd)
    df_gem_eth_usd = pd.read_csv(path_gemini_eth_usd)
    df_gem_ltc_usd = pd.read_csv(path_gemini_ltc_usd)
    df_icobench = pd.read_csv(path_icobench)

    logger.info("Shape of df_bitcoin: {}".format(df_bitcoin.shape))
    logger.info("Shape of df: {}".format(df.shape))
    return df_bitcoin, df, df_test, df_gem_btc_usd, df_gem_eth_usd, df_gem_ltc_usd, df_icobench


def save_x_y(X_train, y_train, X_test):
    logger.info("Saving X_train, y_train and")
    X_train.tofile('data/processed/X_train.np')
    y_train.tofile('data/processed/y_train.np')
    X_test.tofile('data/processed/X_test.np')
