<<<<<<< HEAD
#! /bin/bash
# -*- coding: utf-8 -*-

import logging
import pandas as pd
import numpy as np
from sklearn import preprocessing

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


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
        # Fill all from test set with TEST
        self.df.loc[self.df.success.isna(), 'success'] = "TEST"
        self.df_bitcoin = df_bitcoin
        self.le = preprocessing.LabelEncoder()
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


    def _nan_values_to_string(self, df,column):
        df.loc[df[column].isna(), column] = "NAN"
        return df

    def normalize_categorical_feature(self,df,  column):
        labels = self.le.fit_transform(df[column])
        return labels 
    
    def _add_category(self):
        logger.info("Adding categories_0")
        df_copy = self.df.copy()
        df_copy = self._nan_values_to_string(df_copy, 'categories_0')
        labels = self.normalize_categorical_feature(df_copy, "categories_0")
        df_copy = df_copy.assign(labels_categories_0 = labels)

        self._add_column_to_data_frame(df_copy, "labels_categories_0")
        
    def _add_transaction_count(self):
        """This function adds the feature transaction count to the feature dataset. 
        It will add the feature for train and test set.

        Parameters
        ----------
        train : bool, optional
            Trainset or testset?, by default True
        """
        # Copy DataFrame
        df_copy = self.df.copy()

        df_copy = self._fill_na(
            df_copy, 'transaction_count', 'mean')

        self.df_features = pd.merge(
            self.df_features, df_copy[['OBS_ID', 'transaction_count']])
        assert 'transaction_count' in self.df_features.columns, "No transaction count in df_features!"

    def _add_holder_count(self):
        logger.info("Add function holder count")

        # Copy DataFrame
        df_copy = self.df.copy()

        df_copy = self._fill_na(df_copy, 'holder_count', 'mean')

        # Add for train
        self.df_features = pd.merge(
            self.df_features, df_copy[['OBS_ID', 'holder_count']])
        assert 'transaction_count' in self.df_features.columns, "No transaction count in df_features!"

    def get_X_y(self):
        """This function returns X_train, y_train and X_test.
        These are not the splits for training! This is just for preprocessing both datasets.

        Returns
        -------
        DataFrames
            X_train, y_train, X_test
        """
        df_train = self.df_features.loc[self.df_features.success != 'TEST']
        df_test = self.df_features.loc[self.df_features.success == 'TEST']

        self.X_train = df_train.drop(
            ['success', 'OBS_ID'], axis=1)
        self.y_train = df_train.loc[:, 'success']

        self.X_test = df_test.drop('success', axis=1)

        return self.X_train, self.y_train, self.X_test

    def construct_features(self):
        """This function is the pipeline for adding all features to the dataset
        """
        self._add_transaction_count()
        self._add_holder_count()
        self._add_category()
=======
#! /bin/bash
# -*- coding: utf-8 -*-

import logging
import pandas as pd
import numpy as np
from sklearn import preprocessing

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


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
        self.df = df
        self.df_bitcoin = df_bitcoin
        self.df_test = df_test
        self.le =  preprocessing.LabelEncoder()

        # Create empty dataframe to add features
        self.df_features_train = self.df[['OBS_ID','categories_0', 'success']].copy()
        self.df_features_test = self.df_test[['OBS_ID']].copy()


    def normalize_categorical_feature(self, column):
        self.le.fit(self.df['categories_0'])
        self.le.transform(self.df['categories_0']) 

    def denormalize_categorical_feature(self, column):
        print(column)


    def _add_transaction_count(self, train=True):
        """This function adds the feature transaction count to the feature dataset. 
        It will add the feature for train and test set.

        Parameters
        ----------
        train : bool, optional
            Trainset or testset?, by default True
        """
        logger.info("Adding transaction count for {}".format(
            "train" if train else "test"))
        normalize_categorical_feature(self, "categories_0")

        # Copy DataFrame
        df_copy = self.df.copy() if train else self.df_test.copy()

        # Fill NA values with mean
        df_copy.transaction_count.fillna(
            df_copy.transaction_count.mean(), inplace=True)
        

        if train:
            self.df_features_train = pd.merge(
                self.df_features_train, df_copy[['OBS_ID', 'categories_0' 'transaction_count']])
            assert 'transaction_count' in self.df_features_train.columns, "No transaction count in df_features!"
        else:
            self.df_features_test = pd.merge(
                self.df_features_test, df_copy[['OBS_ID', 'categories_0','transaction_count']])
            assert 'transaction_count' in self.df_features_test.columns, "No transaction count in df_features!"

    def get_X_y(self):
        """This function returns X_train, y_train and X_test.
        These are not the splits for training! This is just for preprocessing both datasets.

        Returns
        -------
        DataFrames
            X_train, y_train, X_test
        """
        self.X_train = self.df_features_train.drop(
            ['success', 'OBS_ID'], axis=1)
        self.y_train = self.df_features_train.loc[:, 'success']

        self.X_test = self.df_features_test

        return self.X_train, self.y_train, self.X_test

    def construct_features(self):
        """This function is the pipeline for adding all features to the dataset
        """
        self._add_transaction_count(train=True)
        self._add_transaction_count(train=False)
>>>>>>> 08ede1a0365feeba9ccdd5929537f2d1a20eaa57
