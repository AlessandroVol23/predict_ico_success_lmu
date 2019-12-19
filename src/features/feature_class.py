#! /bin/bash
# -*- coding: utf-8 -*-

import logging
import pandas as pd
from multiprocessing.pool import ThreadPool, Pool
import numpy as np
from sklearn import preprocessing
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from tqdm import tqdm
import re
from scipy.stats import pearsonr
import datetime as dt

logger = logging.getLogger(__name__)

log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

DATA_FRAME_LENGTH = 5758


class FeatureEngineering(object):

    # Constructor
    def __init__(self, df, df_bitcoin, df_test, df_gem_btc_usd, df_gem_eth_usd, df_gem_ltc_usd, df_icobench):
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
        self.df_gem_btc_usd = df_gem_btc_usd
        self.df_gem_eth_usd = df_gem_eth_usd
        self.df_gem_ltc_usd = df_gem_ltc_usd
        self.df_icobench = df_icobench
        self.dependencies = []

        # Label Encoder
        self.le = preprocessing.LabelEncoder()
        self.url_reg_ex = regex = re.compile(
            r'^(?:http|ftp)s?://'  # http:// or https://
            # domain...
            r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|'
            r'localhost|'  # localhost...
            r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
            r'(?::\d+)?'  # optional port
            r'(?:/?|[/?]\S+)$', re.IGNORECASE)

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
        assert column in self.df_features.columns, "No {} in df_features!".format(
            column)

    def _make_valid_url(self, url):

        result = re.match(self.url_reg_ex, url)

        if result:
            return url
        else:
            return "https://"+url

    def _add_to_base_df(self, column):
        self.df = pd.merge(
            self.df, self.df_features[['OBS_ID', column]])
        assert column in self.df.columns, "No {} in df!".format(
            column)

    def _remove_column_from_data_frame(self, column):
        self.df_features = self.df_features.drop(columns=[column])
        assert column not in self.df_features.columns, "{} is in df_features!".format(
            column)

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

    def _divide_columns(self, column, include_columns, df_copy):
        # subtract all given columns
        for feature in include_columns:
            if include_columns[0] == feature:
                df_copy[column] = df_copy[feature]
                continue
            df_copy[column] = df_copy[column] / df_copy[feature]

        df_copy.loc[df_copy[column] < 0, column] = 0

        return df_copy

    def _calculate_difference(self, column, include_columns, df_copy):
        # subtract all given columns
        for feature in include_columns:
            if include_columns[0] == feature:
                df_copy[column] = df_copy[feature]
                continue
            df_copy[column] = df_copy[column] - df_copy[feature]

        df_copy.loc[df_copy[column] < 0, column] = 0

        return df_copy

    def _transform_numerical_division(self, column, include_columns, na_strategy='set:0'):
        """Divides two or more columns through each other

        Arguments:
            column {String} -- The new features name
            include_columns {Array} -- The features which will be subtracted from each other from left to right
            na_strategy {String} -- A valid na_strategy which is executed before subtraction
        """
        logger.debug(
            "Division of numerical variables for columns {}".format(include_columns))

        # Copy Dataframe
        df_copy = self.df
        # Fill NAs and change dtype to numerical
        for feature in include_columns:
            df_copy = self._execute_na_strategy(df_copy, feature, na_strategy)
            df_copy[feature] = pd.to_numeric(df_copy[feature])

        df_copy = self._divide_columns(column, include_columns, df_copy)

        self._add_column_to_data_frame(df_copy, column)

    def _transform_average_feature(self, column, include_columns=[], na_strategy='median'):
        """Subtracts two or more columns from each other

        Arguments:
            column {String} -- The new features name
            include_columns {Array} -- The features which will be subtracted from each other from left to right
            na_strategy {String} -- A valid na_strategy which is executed before subtraction
        """
        logger.debug(
            "Difference of numerical variables for columns {}".format(include_columns))

        # Copy Dataframe
        df_copy = self.df
        # Fill NAs and change dtype to numerical
        for feature in include_columns:
            df_copy = self._execute_na_strategy(df_copy, feature, na_strategy)
            df_copy[feature] = pd.to_numeric(df_copy[feature])

        df_copy[column] = df_copy[include_columns].mean(axis=1)
        self._add_column_to_data_frame(df_copy, column)

    def _transform_numerical_difference(self, column, include_columns, na_strategy='set:0'):
        """Subtracts two or more columns from each other

        Arguments:
            column {String} -- The new features name
            include_columns {Array} -- The features which will be subtracted from each other from left to right
            na_strategy {String} -- A valid na_strategy which is executed before subtraction
        """
        logger.debug(
            "Difference of numerical variables for columns {}".format(include_columns))

        # Copy Dataframe
        df_copy = self.df
        # Fill NAs and change dtype to numerical
        for feature in include_columns:
            df_copy = self._execute_na_strategy(df_copy, feature, na_strategy)
            df_copy[feature] = pd.to_numeric(df_copy[feature])

        df_copy = self._calculate_difference(column, include_columns, df_copy)

        self._add_column_to_data_frame(df_copy, column)

    def _transform_duration_feature(self, column, include_columns, na_strategy='set:0'):
        """Subtracts two or more columns from each other

        Arguments:
            column {String} -- The new features name
            include_columns {Array} -- The features which will be subtracted from each other from left to right
            na_strategy {String} -- A valid na_strategy which is executed before subtraction
        """
        logger.debug(
            "Difference of numerical variables for columns {}".format(include_columns))

        # Copy Dataframe
        df_copy = self.df
        years = column + '_years'
        months = column + '_months'
        days = column + '_days'

        # Date time to unix timestamp
        for feature in include_columns:
            df_copy[feature] = pd.to_datetime(
                df_copy[feature], infer_datetime_format=True, errors='coerce')
            try:
                df_copy[feature] = df_copy[feature].dt.tz_localize(None)
            except:
                pass

            print(df_copy[feature].head())

        # calculate diffs
        timeDiffs = df_copy[include_columns[0]] - df_copy[include_columns[1]]
        # df_copy[years] = timeDiffs /np.timedelta64(1,'Y')
        # df_copy[months] = timeDiffs /np.timedelta64(1,'M')
        df_copy[column] = timeDiffs / np.timedelta64(1, 'D')

        # fill na values
        # df_copy = self._execute_na_strategy(df_copy, years, na_strategy)
        # df_copy = self._execute_na_strategy(df_copy, months, na_strategy)
        df_copy = self._execute_na_strategy(df_copy, column, na_strategy)

        # self._add_column_to_data_frame(df_copy, years)
        # self._add_column_to_data_frame(df_copy, months)
        self._add_column_to_data_frame(df_copy, column)

    def _add_dependencies(self, dependencies=[]):

        for dep in dependencies:
            self.dependencies.append(dep)

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

    def _rename_column(self, column, rename):
        df_copy = self.df_features.copy()

        df_copy[rename] = df_copy[column]

        self._remove_column_from_data_frame(column)
        self._add_column_to_data_frame(df_copy, rename)

    def _transform_link_binary(self, column):
        try:
            df_link_feature = pd.read_csv('data/external/'+column + '.csv')
            self._add_column_to_data_frame(df_link_feature, column)
        except:
            logger.warn(
                'could not add link binary feature {}, csv file was not found'.format(column))

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

    def get_all_kw_cols(self):
        """
        Get all columns which are associated to KW
        returns: list with kw columns
        """
        cols = list(self.df.columns)
        kws = [s for s in cols if "kw" in s.lower()]
        kws.sort(key=self.natural_keys)
        return kws

    def remove_kw_from_column(self, kws):
        """
        Removes all string chars from KW columns
        returns: list
        """
        new_cols = []
        for i in list(kws):
            new_cols.append(i.replace('KW', ''))

        return new_cols

    def atof(self, text):
        try:
            retval = float(text)
        except ValueError:
            retval = text
        return retval

    def natural_keys(self, text):
        return [self.atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text)]

    def calc_ext_difference(self, amt_weeks: int, df_ext, col_name):
        """
        Function to calculate the differences of calendar weeks and bitcoin price.
        amt_weeks: Number of weeks to go back from last week available
        """
        kws = self.get_all_kw_cols()
        kws.append('OBS_ID')
        df_kws = self.df.loc[:, kws]

        kws_wo_id = set(kws) - set(['OBS_ID'])
        kws_wo_id = list(kws_wo_id)
        kws_wo_id.sort(key=self.natural_keys)

        kws_slice = kws_wo_id[-amt_weeks:]

        grouped_prices_kws = df_ext.groupby(
            'calendar_week').mean()['High']

        new_df = pd.DataFrame(df_kws.OBS_ID)

        for week in kws_slice:

            new_col = col_name + '_' + week
            btc_col = int(re.findall(
                r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', new_col)[0])
            btc_price = grouped_prices_kws[btc_col]
            difference = df_kws.loc[:, week] - btc_price
            new_df.loc[:, new_col] = difference

        return new_df

    def _build_bitcoin_difference(self, amt_weeks: int):
        """
        Function to build difference between bitcoin price and ico price.
        amt_weeks: Number of weeks to go back from last week
        """
        logger.info("Create bitcoin difference feature")
        new_df = self.calc_ext_difference(
            amt_weeks, self.df_gem_btc_usd, 'btc_difference')
        self._add_df_to_feature_df(new_df)

    def _build_eth_difference(self, amt_weeks: int):
        """
        Function to build difference between bitcoin price and ico price.
        amt_weeks: Number of weeks to go back from last week
        """
        logger.info("Create eth difference feature")
        new_df = self.calc_ext_difference(
            amt_weeks, self.df_gem_eth_usd, 'eth_difference')
        self._add_df_to_feature_df(new_df)

    def _build_ltc_difference(self, amt_weeks: int):
        """
        Function to build difference between bitcoin price and ico price.
        amt_weeks: Number of weeks to go back from last week
        """
        logger.info("Create ltc difference feature")
        new_df = self.calc_ext_difference(
            amt_weeks, self.df_gem_ltc_usd, 'ltc_difference')
        self._add_df_to_feature_df(new_df)

    def _build_bitcoin_avg_difference(self):
        logger.info("Build average difference over all weeks")
        df_differences = self.calc_ext_difference(
            39, self.df_gem_btc_usd, 'btc_difference')
        cols = set(df_differences.columns) - set('OBS_ID')
        df_differences_wo_id = df_differences.loc[:, cols]
        mean_per_ico = df_differences_wo_id.mean(axis=1)
        df_differences['mean_difference_btc'] = mean_per_ico
        self._add_column_to_data_frame(df_differences, 'mean_difference_btc')

    def _build_eth_avg_difference(self):
        logger.info("Build average difference over all weeks for eth")
        df_differences = self.calc_ext_difference(
            39, self.df_gem_eth_usd, 'eth_difference')
        cols = set(df_differences.columns) - set('OBS_ID')
        df_differences_wo_id = df_differences.loc[:, cols]
        mean_per_ico = df_differences_wo_id.mean(axis=1)
        df_differences['mean_difference_eth'] = mean_per_ico
        self._add_column_to_data_frame(df_differences, 'mean_difference_eth')

    def _build_ltc_avg_difference(self):
        logger.info("Build average difference over all weeks for ltc")
        df_differences = self.calc_ext_difference(
            39, self.df_gem_ltc_usd, 'ltc_difference')
        cols = set(df_differences.columns) - set('OBS_ID')
        df_differences_wo_id = df_differences.loc[:, cols]
        mean_per_ico = df_differences_wo_id.mean(axis=1)
        df_differences['mean_difference_ltc'] = mean_per_ico
        self._add_column_to_data_frame(df_differences, 'mean_difference_ltc')

    def calc_coeff_kw(self, df_external, col_name):
        kws = self.get_all_kw_cols()
        kws.append('OBS_ID')
        df_kws = self.df.loc[:, kws]

        kws_wo_id = set(kws) - set(['OBS_ID'])
        kws_wo_id = list(kws_wo_id)
        kws_wo_id.sort(key=self.natural_keys)

        grouped_prices_kws = df_external.groupby(
            'calendar_week').mean()['High']
        ext_price = grouped_prices_kws[:39].values

        logger.info("Calculate pearson coefficient of {}".format(col_name))

        for index, row in df_kws.iterrows():
            ico_price = row[kws_wo_id].values
            correlation = pearsonr(ico_price, ext_price)[0]
            df_kws.loc[df_kws.OBS_ID == row.OBS_ID, col_name] = correlation

        return df_kws[['OBS_ID', col_name]]

    def _build_btc_coeff(self):
        df_kws = self.calc_coeff_kw(self.df_gem_btc_usd, 'corr_btc')
        self._add_column_to_data_frame(df_kws, 'corr_btc')

    def _build_eth_coeff(self):
        df_kws = self.calc_coeff_kw(self.df_gem_eth_usd, 'corr_eth')
        self._add_column_to_data_frame(df_kws, 'corr_eth')

    def _build_ltc_coeff(self):
        df_kws = self.calc_coeff_kw(self.df_gem_ltc_usd, 'corr_ltc')
        self._add_column_to_data_frame(df_kws, 'corr_ltc')

    def _build_exist_icobench(self):
        col = 'company_name'
        df_ids = self.df[[col, 'OBS_ID']]
        df_ids[col] = df_ids[col].str.lower()
        df_ids[col] = df_ids[col].str.strip()

        count_exist = 0
        tqdm.write("Creating exist on icobench feature")
        for index, row in tqdm(df_ids.iterrows(), total=df_ids.shape[0]):
            try:
                if self.df_icobench.id.str.contains(row[col]).any():
                    count_exist += 1
                    df_ids.loc[df_ids[col] == row[col],
                               'exist_on_icobench'] = 1
                else:
                    df_ids.loc[df_ids[col] == row[col],
                               'exist_on_icobench'] = 0
            except Exception as e:
                logger.warning("Exception: {}".format(e))

        self._add_column_to_data_frame(df_ids, 'exist_on_icobench')
        logger.info(
            "{} icos were matched with thos on icobench".format(count_exist))

    def _check_meta_information(self, feature, feature_name):
        assert (
            'na_strategy' in feature), "No na_strategy for difference {} provided".format(
                feature_name)
        strategy = feature["na_strategy"]
        assert (
            'columns' in feature), "No columns for difference in feature {} provided".format(feature_name)
        columns = feature["columns"]
        assert (
            len(columns) > 1), "Please provide at least 2 columns for difference {} provided".format(
                feature_name)

        return strategy, columns

    def construct_feature_set(self, features):
        """This function is the pipeline for adding all features to the dataset
        """
        # Iterate through features beforehand for deleting nas
        for feature in features:
            if 'meta' in feature:
                continue

            assert ('column' in feature), "No column key provided"
            feature_name = feature["column"]

            if 'na_strategy' in feature and feature['na_strategy'] == "delete":
                self._delete_na_values(feature_name)

        self._init_df_features()

        # Check dependencies
        for feature in features:
            if 'meta' in feature:
                continue
            if 'dependsOn' in feature:
                feature_name = feature["column"]
                dependencies = feature["dependsOn"]
                assert (
                    len(dependencies) > 0), "Please provide at least 1 dependency for {} ".format(
                        feature_name)
                self._add_dependencies(dependencies)

        # rearange based on dependencies
        features_copy = features.copy()
        for feature in features:
            if 'meta' in feature:
                continue

            feature_name = feature["column"]
            if feature_name in self.dependencies:
                features_copy.remove(feature)
                features_copy.insert(0, feature)

        # Iterating through features and construct feature set
        for feature in features:
            logger.debug("Feature: {}".format(feature))

            if 'meta' in feature:
                feature.pop('meta')
                continue

            if feature['column'] == 'bitcoin_difference':
                amt_weeks = int(feature['amt_weeks'])
                self._build_bitcoin_difference(amt_weeks)
                continue
            elif feature['column'] == 'eth_difference':
                amt_weeks = int(feature['amt_weeks'])
                self._build_eth_difference(amt_weeks)
                continue
            elif feature['column'] == 'ltc_difference':
                amt_weeks = int(feature['amt_weeks'])
                self._build_ltc_difference(amt_weeks)
                continue
            elif feature['column'] == 'bitcoin_avg_difference':
                self._build_bitcoin_avg_difference()
                continue
            elif feature['column'] == 'eth_avg_difference':
                self._build_eth_avg_difference()
                continue
            elif feature['column'] == 'ltc_avg_difference':
                self._build_ltc_avg_difference()
                continue
            elif feature['column'] == 'btc_coeff':
                self._build_btc_coeff()
                continue
            elif feature['column'] == 'eth_coeff':
                self._build_eth_coeff()
                continue
            elif feature['column'] == 'ltc_coeff':
                self._build_ltc_coeff()
                continue
            elif feature['column'] == 'exist_on_icobench':
                self._build_exist_icobench()
                continue

            assert (
                'column' in feature), "No column key provided in feature " + feature
            assert ('type' in feature), "No column type provided"

            feature_type = feature["type"]
            feature_name = feature["column"]

            if feature_type == "categorical":
                assert (
                    'encoder' in feature), "No encoder for categorical feauter {} provided".format(feature_name)

                feauter_encoder = feature["encoder"]
                assert (
                    'na_strategy' in feature), "No na_strategy for numerical feauter {} provided".format(
                    feature_name)
                strategy = feature["na_strategy"]

                if feauter_encoder == "label":
                    self._transform_categorical_variables_label_encoded(
                        feature_name, strategy)
                elif feauter_encoder == "one_hot":
                    self._transform_categorical_variables_one_hot_encoded(
                        feature_name, strategy)
                elif feauter_encoder == "skip":
                    self._transform_categorical_skip_encoded(
                        feature_name, strategy)
                else:
                    raise ValueError("Feauter encoder not recognized")

            elif feature_type == "numerical":
                assert (
                    'na_strategy' in feature), "No na_strategy for categorical feauter {} provided".format(
                    feature_name)

                strategy = feature["na_strategy"]
                self._transform_numerical_variables(feature_name, strategy)

            elif feature_type == "average":
                strategy, columns = self._check_meta_information(
                    feature, feature_name)
                self._transform_average_feature(
                    feature_name, columns, strategy)

            elif feature_type == "difference" or feature_type == "divide":
                strategy, columns = self._check_meta_information(
                    feature, feature_name)

                if feature_type == "difference":
                    self._transform_numerical_difference(
                        feature_name, columns, strategy)
                else:
                    self._transform_numerical_division(
                        feature_name, columns, strategy)
            elif feature_type == "duration":
                strategy, columns = self._check_meta_information(
                    feature, feature_name)
                assert (
                    len(columns) == 2), "Please provide exact 2 columns for duration {} provided".format(
                    feature_name)
                self._transform_duration_feature(
                    feature_name, columns, strategy)

            elif feature_type == "binary":
                self._transform_binary_variables(feature_name)

            elif feature_type == "link":
                self._transform_link_binary(feature_name)
            else:
                raise ValueError('feature type not recognized')

            if "rename" in feature:
                rename = feature["rename"]
                self._rename_column(feature_name, rename)

            if feature_name in self.dependencies:
                name = feature_name
                if "rename" in feature:
                    name = feature["rename"]
                self._add_to_base_df(name)

            if "include" in feature:
                include = feature["include"]
                if include == False:
                    self._remove_column_from_data_frame(feature_name)

        # Check and impolate
        if len(self.to_impute) > 0:
            logger.debug("Start impulating NA values")
            logger.debug("Length of to_impute: {}".format(len(self.to_impute)))
            self._execute_impulate_na()
