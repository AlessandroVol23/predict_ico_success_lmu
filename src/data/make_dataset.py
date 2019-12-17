#! /bin/bash
# -*- coding: utf-8 -*-
import logging
import pandas as pd
import numpy as np
import click
from datetime import datetime

logger = logging.getLogger(__name__)

_COLS_TO_CONVERT = [
    'market_data_current_price_usd',
    'market_data_circulating_supply',
    'market_data_ath_usd',
    'market_data_high_24h_usd',
    'market_data_low_24h_usd',
    'KW1',
    'KW2',
    'KW3',
    'KW4',
    'KW5',
    'KW6',
    'KW7',
    'KW8',
    'KW9',
    'KW10',
    'KW11',
    'KW12',
    'KW13',
    'KW14',
    'KW15',
    'KW16',
    'KW17',
    'KW18',
    'KW19',
    'KW20',
    'KW21',
    'KW22',
    'KW23',
    'KW24',
    'KW25',
    'KW26',
    'KW27',
    'KW28',
    'KW29',
    'KW30',
    'KW31',
    'KW32',
    'KW33',
    'KW34',
    'KW35',
    'KW36',
    'KW37',
    'KW38',
    'KW39',
    'ico_data_total_raised'
]


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

    for col in _COLS_TO_CONVERT:
        df = _replace_convert_float(df, col)

    df = df.assign(id = df.loc[:, 'ï..id'])
    df = df.drop('ï..id', axis=1)

    df = df.rename({'name':'company_name'}, axis=1)

    logger.info("Preprocessing done!")
    return df


def preprocess_bitcoin(df: pd.DataFrame):
    logger.info("Preprocess bitcoin dataset. Shape: {}".format(df.shape))
    logger.info("Build timestamps from milliseconds")
    df['time'] = df.date_in_ms.apply(
        lambda x: datetime.fromtimestamp(x / 1000.0))

    logger.info("Remove all bitcoin prices which are not from 2019")
    df = df.loc[df.time.dt.year == 2019]

    logger.info("Create calendar week.")
    df = df.assign(calendar_week=df.time.dt.week)
    logger.info("End shape of bitcoin dataset: {}".format(df.shape))
    return df


def get_processed_data(path_bitcoin_df='data/raw/1_training_data_sets/1_bitcoin_price_data_set.csv',
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

    df_bitcoin = preprocess_bitcoin(df_bitcoin)
    return df_bitcoin, df, df_test


def get_external_data():
    df_gemin_btc_usd = pd.read_csv('data/external/Gemini_BTCUSD_d.csv')
    df_gemin_eth_usd = pd.read_csv('data/external/Gemini_ETHUSD_d.csv')
    df_gemin_ltc_usd = pd.read_csv('data/external/Gemini_LTCUSD_d.csv')
    df_icobench = pd.read_csv('data/external/ico_bench_ended.csv')
    return df_gemin_btc_usd, df_gemin_eth_usd, df_gemin_ltc_usd, df_icobench


def _save_processed_data(df_bitcoin, df, df_test, df_gem_btc_usd, df_gem_eth_usd, df_gem_ltc_usd, df_icobench):
    df_bitcoin.to_csv('data/processed/df_bitcoin_pp.csv', index=None)
    df.to_csv('data/processed/df_train_pp.csv')
    df_test.to_csv('data/processed/df_test_pp.csv')
    df_gem_btc_usd.to_csv('data/processed/df_gem_btc_usd.csv', index=None)
    df_gem_eth_usd.to_csv('data/processed/df_gem_eth_usd.csv', index=None)
    df_gem_ltc_usd.to_csv('data/processed/df_gem_ltc_usd.csv', index=None)
    df_icobench.to_csv('data/processed/df_icobench.csv', index=None)


def preprocess_external_data(df_btc, df_eth, df_ltc):
    def preprocess_times(df):
        df['Date'] = pd.to_datetime(df.Date)
        df = df.loc[df.Date.dt.year == 2019]
        df = df.assign(calendar_week=df.Date.dt.week)
        return df

    df_btc_pp = preprocess_times(df_btc)
    df_eth_pp = preprocess_times(df_eth)
    df_ltc_pp = preprocess_times(df_ltc)

    return df_btc_pp, df_eth_pp, df_ltc_pp


@click.command()
def main():
    df_bitcoin, df, df_test = get_processed_data()
    df_gemin_btc_usd, df_gemin_eth_usd, df_gemin_ltc_usd, df_icobench = get_external_data()
    df_btc_pp, df_eth_pp, df_ltc_pp = preprocess_external_data(
        df_gemin_btc_usd, df_gemin_eth_usd, df_gemin_ltc_usd)
    _save_processed_data(df_bitcoin, df, df_test,
                         df_btc_pp, df_eth_pp, df_ltc_pp, df_icobench)


if __name__ == "__main__":
    main()
