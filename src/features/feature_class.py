#! /bin/bash
# -*- coding: utf-8 -*-

import logging
import pandas as pd
import numpy as np
from sklearn import preprocessing

logger = logging.getLogger(__name__)
# log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
# logging.basicConfig(level=logging.INFO, format=log_fmt)


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

        logger.debug("Start filling NA values in {} with strategy".format(column, strategy))
        
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

            else:
                raise ValueError("Unrecognized na strategy for {column}")
            df[column].fillna(to_fill, inplace=True)

        logger.debug("Filled NA values")
        return df

    def _add_column_to_data_frame(self, df, column):
        self.df_features = pd.merge(
            self.df_features, df[['OBS_ID', column]])
        assert column in self.df_features.columns, "No {column} in df_features!"

    def _label_encode_categorical_feature(self, df,  column):
        labels = self.le.fit_transform(df[column])
        return labels

    def _one_hote_encoder(self, df, column):
        categorical_data = df[column].values.reshape(-1, 1)
        onehot_encoded = self.enc.fit_transform(categorical_data).toarray()
        return onehot_encoded

    def _transform_numerical_variables(self, column, na_strategy='mean'):
        logger.debug("Transform numerical variable for column {}".format(column))

        # Copy Dataframe
        df_copy = self.df

        # Fill NAs
        df_copy = self._fill_na(df_copy, column, na_strategy)

        self._add_column_to_data_frame(df_copy, column)

    def _transform_categorical_variables_label_encoded(self, column, na_strategy='set:NAN'):
        logger.debug(
            "Transform categorical variable to label encoding for column {}".format(column))

        label_name = "labels_" + column
        self.label_dict[column] = label_name
        # Copy Dataframe
        df_copy = self.df.copy()

        # Fill NAs
        df_copy = self._fill_na(df_copy, column, na_strategy)

        # Transform labels
        labels = self._label_encode_categorical_feature(df_copy, column)
        dictonary = {self.label_dict[column]: labels}
        df_copy = df_copy.assign(**dictonary)

        self._add_column_to_data_frame(df_copy, self.label_dict[column])

    def _transform_categorical_variables_one_hot_encoded(self, column,na_strategy='set:NAN'):
        logger.debug(
            "Transform categorical variable to one hot encoded for column {}".format(column))
        # Copy Dataframe
        df_copy = self.df.copy()

        # Fill NAs
        df_copy = self._fill_na(df_copy, column, na_strategy)
        label_name = column + "_"

        # Tansform one hot encoded
        labels = self._one_hote_encoder(df_copy, column)
        dfOneHot = pd.DataFrame(
            labels, columns=[label_name + str(int(i)) for i in range(labels.shape[1])])

        # Concat to
        df_copy = df_copy.join(dfOneHot)

        # TODO
        # Currently little tricky, as we iterrate over the onHoteEncoded data but provide the df_copy dataframe as parameter
        # maybe we should just concatinate the one_hote_encoded datafarame to the object dataframe in a seperate function ??
        for col in dfOneHot.columns:
            self._add_column_to_data_frame(df_copy, col)

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
        logger.info("X_train shape: {}".format(self.X_train.shape))
        logger.info("y_train shape: {}".format(self.y_train.shape))
        logger.info("X_test shape: {}".format(self.X_test.shape))

        return self.X_train, self.y_train, self.X_test

            

    def construct_feature_set(self,featuers):
        """This function is the pipeline for adding all features to the dataset
        """
        for feature in featuers:
            assert ('column' in feature), "No column key provided"
            assert("type" in feature), "No column type provided"

            feature_type = feature["type"]
            feature_name = feature["column"]

            if feature_type == "categorical":
                assert ('encoder' in feature), "No encoder for categorical feauter {feature_name} provided"
                feauter_encoder = feature["encoder"]

                if feauter_encoder == "label":
                   self._transform_categorical_variables_label_encoded(feature_name)
                elif feauter_encoder == "one_hot":
                   self._transform_categorical_variables_one_hot_encoded(feature_name)
                else: 
                    raise ValueError("Feauter encoder not recognized")

            elif feature_type == "numerical":
                assert ('na_strategy' in feature), "No na_strategy for categorical feauter {feature_name} provided"
                strategy = feature["na_strategy"]
                self._transform_numerical_variables(feature_name, strategy)
            else:
                raise ValueError('feature type not recognized')

