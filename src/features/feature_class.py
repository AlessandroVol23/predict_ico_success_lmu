#! /bin/bash
# -*- coding: utf-8 -*-

import logging
import pandas as pd
import numpy as np
from sklearn import preprocessing

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

_categorical_featuers = [
    'categories_0',
    'country_origin',
    'ico_data_country_origin',
    'ico_data_hardcap_currency',
    'ico_data_softcap_currency',
    'ico_data_total_raised_currency',
    'ico_data_country_origin'
]

_numerical_features = [
    'transaction_count',
    'holder_count',
    'market_data_current_price_usd',
    'market_data_ath_usd',
    'market_data_total_supply',
    'market_data_circulating_supply',
    'market_data_high_24h_usd',
    'market_data_low_24h_usd',
    'public_interest_stats_bing_matches',
    'community_data_facebook_likes',
    'community_data_twitter_followers',
    'community_data_reddit_subscribers',
    'developer_data_stars',
    'developer_data_subscribers',
    'developer_data_total_issues',
    'developer_data_closed_issues',
    'developer_data_pull_requests_merged',
    'developer_data_commit_count_4_weeks',
    'ico_data_hardcap_amount',
    'ico_data_softcap_amount']


class FeatureEngineering(object):

    # Constructor
    def __init__(self, df, df_bitcoin, df_test):
        """Constructor for class FeatureEngineering

        Parameters
        ----------
        df : DataFrame
            Train DataFrame
        df_bitcoin : DataFrame
            Bitcoin DataFrame
        df_test : DataFrame
            Test DataFrame
        """

        self.df = pd.concat([df, df_test])
        assert len(self.df) == 5758, "Length has to be 5758, check conattanation!"
        # Fill all from test set with TEST
        self.df.loc[self.df.success.isna(), 'success'] = "TEST"
        self.df_bitcoin = df_bitcoin

        # Label Encoder
        self.le = preprocessing.LabelEncoder()

        # One Hot Encoder
        self.enc = preprocessing.OneHotEncoder(handle_unknown='ignore')

        self.label_dict = {}
        # Create empty dataframe to add features
        self.df_features = self.df[['OBS_ID', 'success']].copy()

    def _fill_na(self, df_in, column, strategy):
        """Function to fill NA values

        Arguments:
            df_in {DataFrame} -- DataFrame to fill NA values
            column {str} -- Column to fill
            strategy {str} -- Mean / Median so far

        Returns:
            DataFrame -- DataFrame with filled NA values
        """
        logger.info("Start filling NA values in {}".format(column))
        df = df_in.copy()

        if strategy == 'mean':
            to_fill = df[column].mean()
        elif strategy == 'median':
            to_fill = df[column].median()

        logger.info("Found {} NA values in column {}".format(
            df[column].isna().sum(), column))

        df[column].fillna(to_fill, inplace=True)

        logger.info("Filled NA values")
        return df

    def _add_column_to_data_frame(self, df, column):
        self.df_features = pd.merge(
            self.df_features, df[['OBS_ID', column]])
        assert column in self.df_features.columns, "No {column} in df_features!"

    def _nan_values_to_string(self, df, column):
        df.loc[df[column].isna(), column] = "NAN"
        return df

    def normalize_categorical_feature(self, df,  column):
        labels = self.le.fit_transform(df[column])
        return labels

    def one_hote_encoder(self, df, column):
        categorical_data = df[column].values.reshape(-1, 1)
        onehot_encoded = self.enc.fit_transform(categorical_data).toarray()
        return onehot_encoded

    def _add_category(self):
        logger.info("Adding categories_0")
        df_copy = self.df.copy()
        df_copy = self._nan_values_to_string(df_copy, 'categories_0')
        labels = self.normalize_categorical_feature(df_copy, "categories_0")
        df_copy = df_copy.assign(labels_categories_0=labels)

        self._add_column_to_data_frame(df_copy, "labels_categories_0")

    def _transform_numerical_variables(self, column, na_strategy='mean'):
        logger.info("Transform numerical variable for column {}".format(column))

        # Copy Dataframe
        df_copy = self.df

        # Fill NAs
        df_copy = self._fill_na(df_copy, column, na_strategy)

        self._add_column_to_data_frame(df_copy, column)

    def _transform_categorical_variables_label_encoded(self, column):
        logger.info(
            "Transform categorical variable to label encoding for column {}".format(column))

        label_name = "labels_" + column
        self.label_dict[column] = label_name
        # Copy Dataframe
        df_copy = self.df.copy()

        # Fill NAs
        df_copy = self._nan_values_to_string(df_copy, column)

        # Transform labels
        labels = self.normalize_categorical_feature(df_copy, column)
        dictonary = {self.label_dict[column]: labels}
        df_copy = df_copy.assign(**dictonary)

        self._add_column_to_data_frame(df_copy, self.label_dict[column])

    def _transform_categorical_variables_one_hot_encoded(self, column):
        logger.info(
            "Transform categorical variable to one hot encoded for column {}".format(column))
        # Copy Dataframe
        df_copy = self.df.copy()

        # Fill NAs
        df_copy = self._nan_values_to_string(df_copy, column)
        label_name = column + "_"

        # Tansform one hot encoded
        labels = self.one_hote_encoder(df_copy, column)
        dfOneHot = pd.DataFrame(
            labels, columns=[label_name + str(int(i)) for i in range(labels.shape[1])])

        # Concat to
        df_copy = df_copy.join(dfOneHot)

        # TODO
        # Currently little tricky, as we iterrate over the onHoteEncoded data but provide the df_copy dataframe as parameter
        # maybe we should just concatinate the one_hote_encoded datafarame to the object dataframe in a seperate function ??
        for col in dfOneHot.columns:
            self._add_column_to_data_frame(df_copy, col)

    def _add_holder_count(self):
        logger.info("Add function holder count")

        # Copy DataFrame
        df_copy = self.df.copy()

        df_copy = self._fill_na(df_copy, 'holder_count', 'mean')

        # Add for train
        self.df_features = pd.merge(
            self.df_features, df_copy[['OBS_ID', 'holder_count']])
        assert 'holder_count' in self.df_features.columns, "No holder_count in df_features!"

    def get_X_y(self):
        """This function returns X_train, y_train and X_test.
        These are not the splits for training! This is just for preprocessing both datasets.

        Returns
        -------
        DataFrames
            X_train, y_train, X_test
        """
        df_train = self.df_features.loc[self.df_features.success != 'TEST']
        logger.info("DF_train shape: {}".format(df_train.shape))
        df_test = self.df_features.loc[self.df_features.success == 'TEST']
        logger.info("df_test shape: {}".format(df_train.shape))

        self.X_train = df_train.drop(
            ['success', 'OBS_ID'], axis=1)
        self.y_train = df_train.loc[:, 'success'].values.astype(int)
        logger.info("Y_train unique values: {}".format(
            np.unique(self.y_train, return_counts=True)))

        self.X_test = df_test.drop('success', axis=1)

        # self.X_train = self.X_train.values
        # self.y_train = self.y_train.values.astype(int)
        logger.info("X_train shape: {}".format(self.X_train.shape))
        logger.info("y_train shape: {}".format(self.y_train.shape))
        logger.info("X_test shape: {}".format(self.X_test.shape))

        return self.X_train, self.y_train, self.X_test

    def construct_numerical_featuers(self):
        """This function is the pipeline for adding all numerical features to the dataset
        """

        for col in _numerical_features:
            self._transform_numerical_variables(col)

    def construct_cateogrical_features_label_encoded(self):
        """This function is the pipeline for adding all label encoded categorical featuers to the dataset
        """

        for col in _categorical_featuers:
            self._transform_categorical_variables_label_encoded(col)

    def construct_cateogrical_features_one_hot_encoded(self):
        """This function is the pipeline for adding all label encoded categorical featuers to the dataset
        """

        for col in _categorical_featuers:
            self._transform_categorical_variables_one_hot_encoded(col)

    def construct_features(self):
        """This function is the pipeline for adding all features to the dataset
        """
        self.construct_numerical_featuers()
        self.construct_cateogrical_features_one_hot_encoded()
