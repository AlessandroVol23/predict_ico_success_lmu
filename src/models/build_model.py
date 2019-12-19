import click
from src.models.fit_model import FittingModel
from src.models.light_gbm import LightGbmModel
from src.models.catboost_model import CatBoostModel
import logging
import json
import pandas as pd
import numpy as np
from src.utils import read_feature_meta
from time import time
import datetime
from catboost import EFstrType
from src.models.utils import read_upsampling_feature_set

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

training_models = [
    CatBoostModel,
    LightGbmModel
]
training_models = [
    LightGbmModel
]


class BuildModel(object):

    def __init__(self):
        pass

    def _read_result_json(self):
        with open("data/results/result.json", "r") as f:
            return (json.load(f))

    def _read_result_csv(self):
        try:
            df = pd.read_csv('data/results/result.csv')
        except FileNotFoundError:
            logger.info("File wasn't found. Create new one")
            df = pd.DataFrame()
        return df

    def _write_result_json(self, result):
        with open("data/results/result.json", "w") as f:
            json.dump(result, f)

    def _write_result_csv(self, result, result_ser):
        df = result.append(result_ser, ignore_index=True)
        df.to_csv('data/results/result.csv', index=None)

    def _get_submission_number(self):
        with open("SUBMISSION_NUMBER", "r") as f:
            return f.readline()

    def _increment_submission_number(self, current_number=0):
        new_build_number = int(current_number) + 1
        with open("SUBMISSION_NUMBER", "w") as f:
            f.write(str(new_build_number))

    def _create_evaluation_file(self, test_ids, sub_preds_abs, next_submission_number, increment=True):
        if increment:
            self._increment_submission_number(next_submission_number)

        df_submission = pd.DataFrame(
            [test_ids.values, sub_preds_abs]).transpose()
        df_submission.columns = ['OBS_ID', 'success']
        df_submission['OBS_ID'] = df_submission.OBS_ID.astype(int)
        df_submission['success'] = df_submission.success.astype(int)
        fileName = 'data/submissions/submission' + next_submission_number + '.csv'
        df_submission.to_csv(fileName, index=None)
        logger.info("Write submission file to: {}".format(fileName))

    def _write_results(self, feature_set_meta, feature_set_number, mean_mcc, model_name, next_submission_number,
                       hyperparam):
        feature_sets = read_feature_meta()
        if (feature_set_number in feature_sets):
            feature_set = feature_sets[feature_set_number]
        else:
            feature_sets = read_feature_meta(True)
            feature_set = feature_sets[feature_set_number]

        upsampling = read_upsampling_feature_set(
            feature_set_meta, feature_set_number)

        result = self._read_result_csv()

        result_ser = pd.Series({
            'timestamp': datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f"),
            'feature_set': feature_set_number,
            'features': feature_set,
            'model_name': model_name,
            'submission_number': next_submission_number,
            'mcc_cv': mean_mcc,
            'upsampling': upsampling,
            'submission_score': 'TO_FILL',
            'hp_iterations': hyperparam.get('iterations', 'NA'),
            'hp_early_stopping_rounds': hyperparam.get('early_stopping_rounds', 'NA'),
            'hp_n_estimators': hyperparam.get('n_estimators', 'NA'),
            'hp_learning_rate': hyperparam.get('learning_rate', 'NA'),
            'hp_loss_function': hyperparam.get('loss_function', 'NA'),
            'hp_use_best_model': hyperparam.get('use_best_model', 'NA')
        })
        self._write_result_csv(result, result_ser)

    def _read_categorical_features(self, feature_set_meta, feature_set_key):
        feature_set = feature_set_meta[feature_set_key]
        categorical_features = []
        for feature in feature_set:
            if 'meta' in feature:
                continue
            feature_name = feature["column"]
            feature_type = feature["type"]
            if feature_type == "categorical":
                categorical_features.append(feature_name)

        return categorical_features

    def train_model(self, feature_set_key, modelName=""):
        """takes the list of models and fits them with cross validation"""

        for current_model_class in training_models:

            # Init a model class inheritated from BaseModel class
            current_model = current_model_class()
            if modelName != "" and modelName != current_model.get_name():
                continue
            logger.info(
                "Building model with feature set {}".format(feature_set_key))

            feature_set_meta = read_feature_meta()

            upsampling = read_upsampling_feature_set(
                feature_set_meta, feature_set_key)

            categorical_features = self._read_categorical_features(
                feature_set_meta, feature_set_key)

            # Fitting model that trains and cross validates, takes the underlying model to train as a param
            fitting_model = FittingModel(
                feature_set_key, current_model, categorical_features, upsample=upsampling)

            # Get values from fitting model
            mean_mcc = fitting_model.cross_validation()

            test_ids, sub_preds_abs = fitting_model.get_values()

            next_submission_number = self._get_submission_number()
            self._create_evaluation_file(test_ids, sub_preds_abs,
                                         next_submission_number, True)

            # get name and params from underlying model
            model_name = current_model.get_name()
            hyperparams = current_model.get_params()

            self._write_results(feature_set_meta, feature_set_key, mean_mcc, model_name,
                                'No Submission', hyperparams)

            # Workaround till we have function to read in optimized hyperparams
            if model_name == 'catboost':
                hp = {
                    "bagging_temperature": 1.0,
                    "border_count": 121,
                    "depth": 4,
                    "iterations": 802,
                    "l2_leaf_reg": 30,
                    "learning_rate": 0.4476540650629794,
                    "random_strength": 10.0,
                    "scale_pos_weight": 0.9494114772362018
                }
                final_model = current_model_class(hp)
            else:
                final_model = current_model_class()

            fitting_model = FittingModel(
                feature_set_key, final_model, categorical_features, upsample=upsampling)

            logger.info("Create final model.")
            fitting_model.train_final_model()
            preds_test = fitting_model.predict_test_set()
            preds_test_abs = preds_test.argmax(axis=1)
            next_submission_number = self._get_submission_number()
            fitting_model.save_current_model()
            fitting_model.save_feature_importance()
            fitting_model.save_feature_importance("LossFunctionChange", preds_test_abs)
            self._create_evaluation_file(fitting_model.test_ids, preds_test_abs,
                                         next_submission_number, True)


@click.command()
@click.argument('feature_set_key')
@click.argument('model_name')
def main(feature_set_key, model_name):
    build_model = BuildModel()
    build_model.train_model(feature_set_key, model_name)


if __name__ == "__main__":
    main()
