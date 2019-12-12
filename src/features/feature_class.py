#! /bin/bash
# -*- coding: utf-8 -*-

import logging
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

logger = logging.getLogger(__name__)


log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

DATA_FRAME_LENGTH = 5758


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

        self.df = pd.concat([df, df_test], sort=True)
        assert len(
            self.df) == DATA_FRAME_LENGTH, "Length has to be 5758, check conattanation!"
        # Fill all from test set with TEST
        self.df.loc[self.df.success.isna(), 'success'] = "TEST"
        self.df_bitcoin = df_bitcoin

        # Label Encoder
        self.le = preprocessing.LabelEncoder()

        # One Hot Encoder
        self.enc = preprocessing.OneHotEncoder(handle_unknown='ignore')

        self.label_dict = {}

        # List for saving columns to impute
        self.to_impute = []

    def _init_df_features(self):
        # Create empty dataframe to add features
        self.df_features = self.df[['OBS_ID', 'success']].copy()
        self.df_feature_length = len(self.df_features)

    def _delete_na_values(self, column):
        """This function removes all na rows from the original dataframe for a given column

        Returns
        -------
        """
        logger.debug(
            "Removing NA values for feature {} in base dataframe".format(column))
        logger.debug("Old shape of df {}".format(self.df.shape))
        self.df = self.df.dropna(subset=[column])
        logger.debug(
            "New shape of df after dropping NA rows {}".format(self.df.shape))

    def _execute_impulate_na(self):
        imp = IterativeImputer(max_iter=10, random_state=123)

        features = list(set(self.df_features.columns) -
                        set(['success', 'OBS_ID']))

        imp.fit(self.df_features[features])

        for col in self.to_impute:
            logger.debug("Impute {} now".format(col))
            logger.debug("{} has {} empty entries".format(
                col, self.df_features[col].isna().sum()))
            index_of_col = features.index(col)

            na_vals = self.df_features.loc[self.df_features[col].isna(
            ), features]
            imputed = imp.transform(na_vals)
            self.df_features.loc[self.df_features[col].isna(),
                                 col] = imputed[:, index_of_col]

    def _execute_na_strategy(self, df_in, column, strategy):
        """Function to fill NA values

        Arguments:
            df_in {DataFrame} -- DataFrame to fill NA values
            column {str} -- Column to fill
            strategy {str} -- Mean / Median so far

        Returns:
            DataFrame -- DataFrame with filled NA values
        """

        logger.debug(
            "Start filling NA values in {} with strategy {}".format(column, strategy))

        df = df_in.copy()

        logger.debug("Found {} NA values in column {}".format(
            df[column].isna().sum(), column))

        if strategy.find(":") != -1:
            strat = strategy.split(":")
            if strat[0] == "set":
                df.loc[df[column].isna(), column] = strat[1]
            else:
                raise ValueError("Unrecognized command strategy for {column}")

        else:
            if strategy == 'mean':
                to_fill = df[column].mean()

            elif strategy == 'median':
                to_fill = df[column].median()
            elif strategy == 'min':
                to_fill = df[column].min()
            elif strategy == 'max':
                to_fill = df[column].max()
            elif strategy == 'False':
                to_fill = False
            elif strategy == "delete":
                return df
            elif strategy == "impute":
                # Fill temporary with np.nan and handle later
                if df[column].isna().sum() > 0:
                    self.to_impute.append(column)
                to_fill = np.nan
            else:
                raise ValueError("Unrecognized na strategy for {column}")
            df[column].fillna(to_fill, inplace=True)

        logger.debug("Filled NA values")
        return df

    def _add_column_to_data_frame(self, df, column):
        self.df_features = pd.merge(
            self.df_features, df[['OBS_ID', column]])
        assert column in self.df_features.columns, "No {column} in df_features!"

    def _add_df_to_feature_df(self, df):
        """Adds a DataFrame based on OBS_ID to feature df

        Arguments:
            df {DataFrame} -- DataFrame to join
        """
        assert 'OBS_ID' in list(df.columns), "No OBS_ID in df!"
        self.df_features = pd.merge(self.df_features, df)

    def _label_encode_categorical_feature(self, df, column):
        labels = self.le.fit_transform(df[column])
        return labels

    def _one_hote_encoder(self, df, column):
        one_hot = pd.get_dummies(df[column], prefix=column)
        df_one_hot_with_id = pd.concat([df[['OBS_ID']], one_hot], axis=1)
        # self.df_features = pd.merge(self.df_features, df_ohe_with_ide)
        assert len(
            df) == self.df_feature_length, "Length is wrong! One Hot Encoding failed"
        return df_one_hot_with_id

    def _transform_numerical_variables(self, column, na_strategy='mean'):
        logger.debug(
            "Transform numerical variable for column {}".format(column))

        # Copy Dataframe
        df_copy = self.df

        # Fill NAs
        df_copy = self._execute_na_strategy(df_copy, column, na_strategy)

        self._add_column_to_data_frame(df_copy, column)

    def _transform_binary_variables(self, column, na_strategy='set:0'):
        logger.debug("Transform binary variable for column {}".format(column))

        # Copy Dataframe
        df_copy = self.df

        # Fill NAs
        df_copy = self._execute_na_strategy(df_copy, column, na_strategy)
        # df_copy[column][df_copy[column] != '0'] = 1
        df_copy.loc[df_copy[column] != '0', column] = 1

        df_copy[column] = df_copy[column].astype(int)

        self._add_column_to_data_frame(df_copy, column)

    def _transform_categorical_variables_label_encoded(self, column, na_strategy='set:NAN'):
        logger.debug(
            "Transform categorical variable to label encoding for column {}".format(column))

        label_name = "labels_" + column
        self.label_dict[column] = label_name
        # Copy Dataframe
        df_copy = self.df.copy()

        # Fill NAs
        df_copy = self._execute_na_strategy(df_copy, column, na_strategy)

        # Transform labels
        labels = self._label_encode_categorical_feature(df_copy, column)
        dictonary = {self.label_dict[column]: labels}
        df_copy = df_copy.assign(**dictonary)

        self._add_column_to_data_frame(df_copy, self.label_dict[column])

    def _transform_categorical_variables_one_hot_encoded(self, column, na_strategy='set:NAN'):
        logger.debug(
            "Transform categorical variable to one hot encoded for column {}".format(column))
        # Copy Dataframe
        df_copy = self.df.copy()
        # Fill NAs
        df_copy = self._execute_na_strategy(df_copy, column, na_strategy)

        label_name = column + "_"
        # Tansform one hot encoded
        df_ohe_id = self._one_hote_encoder(df_copy, column)

        self._add_df_to_feature_df(df_ohe_id)


    def _transform_categorical_skip_encoded(self, column, na_strategy='set:NAN'):
        logger.debug(
            "Transform categorical variable to one hot encoded for column {}".format(column))
        # Copy Dataframe
        df_copy = self.df.copy()
        # Fill NAs
        df_copy = self._execute_na_strategy(df_copy, column, na_strategy)

        self._add_column_to_data_frame(df_copy, column)

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
        self.y_train = df_train.loc[:, 'success'].values.astype(int)

        self.X_test = df_test.drop('success', axis=1)

        # self.X_train = self.X_train.values
        # self.y_train = self.y_train.values.astype(int)
        logger.debug("X_train shape: {}".format(self.X_train.shape))
        logger.debug("y_train shape: {}".format(self.y_train.shape))
        logger.debug("X_test shape: {}".format(self.X_test.shape))

        return self.X_train, self.y_train, self.X_test

    def construct_feature_set(self, featuers):
        """This function is the pipeline for adding all features to the dataset
        """
        # Iterate through features beforehand for deleting nas
        for feature in featuers:
            if 'meta' in feature:
                continue

            assert ('column' in feature), "No column key provided"
            feature_name = feature["column"]

            if 'na_strategy' in feature and feature['na_strategy'] == "delete":
                self._delete_na_values(feature_name)

        self._init_df_features()

        # Iterating through features and construct feature set
        for feature in featuers:
            if 'meta' in feature:
                feature.pop('meta')
                continue

            assert (
                'column' in feature), "No column key provided in feature " + feature
            assert ('type' in feature), "No column type provided"

            feature_type = feature["type"]
            feature_name = feature["column"]

            if feature_type == "categorical":
                assert (
                    'encoder' in feature), "No encoder for categorical feauter {feature_name} provided"

                feauter_encoder = feature["encoder"]
                assert (
                    'na_strategy' in feature), "No na_strategy for numerical feauter {feature_name} provided"
                strategy = feature["na_strategy"]

                if feauter_encoder == "label":
                    self._transform_categorical_variables_label_encoded(
                        feature_name, strategy)
                elif feauter_encoder == "one_hot":
                    self._transform_categorical_variables_one_hot_encoded(
                        feature_name, strategy)
                elif feauter_encoder == "skip":
                    self._transform_categorical_skip_encoded(feature_name, strategy)
                else:
                    raise ValueError("Feauter encoder not recognized")

            elif feature_type == "numerical":
                assert (
                    'na_strategy' in feature), "No na_strategy for categorical feauter {feature_name} provided"
                strategy = feature["na_strategy"]
                self._transform_numerical_variables(feature_name, strategy)

            elif feature_type == "binary":
                self._transform_binary_variables(feature_name)
            else:
                raise ValueError('feature type not recognized')

        # Check and impolate
        if len(self.to_impute) > 0:
            logger.debug("Start impulating NA values")
            logger.debug("Length of to_impute: {}".format(len(self.to_impute)))
            self._execute_impulate_na()
