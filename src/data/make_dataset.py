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


def preprocess_dtypes(df_in):
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

    # Copy DataFrame -> If not you edit the original one in the memory
    df = df_in.copy()

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
    return df_bitcoin, df, df_test
