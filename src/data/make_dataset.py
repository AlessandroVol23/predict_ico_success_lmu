#! /bin/bash
# -*- coding: utf-8 -*-
import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def read_in_data(path_bitcoin_df='data/raw/1_training_data_sets/1_bitcoin_price_data_set.csv',
                 path_training_df='data/raw/1_training_data_sets/1_training_data.csv',
                 path_test_df='data/raw/2_classification_data.csv'):
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


def clean_data(df_in):
    """This function cleans data and removes errornous data.

    Parameters
    ----------
    df_in : DataFrame
        DataFrame to be cleaned

    ReturnsFeatureEngineering
    -------
    DataFrame
        Cleaned DataFrame
    """
    df = df_in.copy()

    return df


def preprocess(df_in):
    """This function preprocessed and changes all columns to the right dtype.

    Parameters
    ----------
    df_in : DataFrame
        Original DataFrame

    Returns
    -------
    DataFrame
        Preprocessed DataFrame
    """
    def _replace_convert_float(df, column, to_replace=',', replace_with='.', convert_to='float'):
        logger.info("Replacing {} ".format(column))
        df[column] = df[column].astype(str)
        df[column] = df[column].apply(lambda x: x.replace(
            to_replace, replace_with)).astype(convert_to)
        return df

    logger.info("Start preprocessing dataframe")
    # Copy DataFrame -> If not you edit the original one in the memory
    df = df_in.copy()

    # market_data_current_price_usd
    df = _replace_convert_float(df, 'market_data_current_price_usd')

    # market_data_ath_usd
    df = _replace_convert_float(df, 'market_data_ath_usd')

    # market_data_circulating_supply
    df = _replace_convert_float(df, 'market_data_circulating_supply')

    logger.info("Preprocessing done!")
    return df


def get_preprocessed_datasets(path_bitcoin_df='data/raw/1_training_data_sets/1_bitcoin_price_data_set.csv',
                              path_training_df='data/raw/1_training_data_sets/1_training_data.csv',
                              path_test_df='data/raw/2_classification_data.csv'):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    logger.info('making final data sets from raw data')
    df_bitcoin, df, df_test = read_in_data()

    # Concat for preprocessing
    df = pd.concat([df, df_test])
    df.loc[df.success.isna(), 'success'] = "TEST"

    df = preprocess(df)

    # Split into df and df_test again
    df_test = df.loc[df.success == "TEST", :"KW39"]
    df = df.loc[df.success != "TEST"]
    assert len(df) == 4757, "Shape of DF has to be 4757"
    assert len(df_test) == 1001, "Shape of DF test has to be 1001"
    return df_bitcoin, df, df_test
