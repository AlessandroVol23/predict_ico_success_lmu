import click
from light_gbm import LightGbmModel
import logging
import json
import numpy as np
import pandas as pd
from src.utils import read_feature_meta
from time import time
import os

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


class BuildModel(object):

    def __init__(self):
        pass

    def _read_result_json(self):
        with open("data/results/result.json", "r") as f:
            return (json.load(f))

    def _write_result_json(self, result):
        with open("data/results/result.json", "w") as f:
            json.dump(result, f)

    def _get_submission_number(self):
        with open("SUBMISSION_NUMBER", "r") as f:
            return f.readline()

    def _increment_submission_number(self, current_number=0):
        new_build_number = int(current_number) + 1
        logger.info('New submission number is now: {}'.format(new_build_number))
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

    def _write_results(self, feature_set_meta, feature_set_number, mean_mcc, model_name, next_submission_number, hyperparam):
        feature_sets = read_feature_meta()
        if (feature_set_number in feature_sets):
            feature_set = feature_sets[feature_set_number]
        else:
            feature_sets = read_feature_meta(True)
            feature_set = feature_sets[feature_set_number]

        result = self._read_result_json()

        metrics = {}
        metrics['mcc'] = mean_mcc

        result.append({
            'timestamp': str(time()),
            'feature_set': feature_set_number,
            'features': feature_set,
            'metrics': metrics,
            'model_name': model_name,
            'hyperparam': hyperparam,
            'submission': next_submission_number
        })
        self._write_result_json(result)

    def read_upsampling_feature_set(self, feature_set_meta, feature_set_key):
        feature_set = feature_set_meta[feature_set_key]
        for feature in feature_set:
            if 'meta' in feature:
                try:
                    upsampling = float(feature['meta']['upsampling'])
                    return upsampling
                except ValueError:
                    logger.warning(
                        "Won't upsample because no float value was provided!")

    def train_model(self, feature_set_key):
        logger.info(
            "Building model with feature set {}".format(feature_set_key))
        feature_set_meta = read_feature_meta()
        upsampling = self.read_upsampling_feature_set(
            feature_set_meta, feature_set_key)

        model = LightGbmModel(feature_set_key, upsample=upsampling)

        next_submission_number = self._get_submission_number()

        mean_mcc, hyperparam = model.cross_validation()
        test_ids, sub_preds_abs = model.get_values()
        model_name = model.get_name()

        self._create_evaluation_file(test_ids, sub_preds_abs,
                                     next_submission_number, True)

        self._write_results(feature_set_meta, feature_set_key, mean_mcc, model_name,
                            next_submission_number, hyperparam)


@click.command()
@click.argument('feature_set_key')
def main(feature_set_key):
    build_model = BuildModel()
    build_model.train_model(feature_set_key)


if __name__ == "__main__":
    main()
