import click
from light_gbm import LightGbmModel
import logging
import json
import numpy as np
import pandas as pd
from src.utils import read_feature_meta
from time import time

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

def _read_result_json():
    with open("data/results/result.json", "r") as f:
        return (json.load(f))

def _write_result_json(result):
    with open("data/results/result.json", "w") as f:
        json.dump(result, f)

def _get_submission_number():
    with open("SUBMISSION_NUMBER", "r") as f:
        return f.readline()

def _increment_submission_number(current_number=0):
    new_build_number = int(current_number) + 1
    logger.info('New submission number is now: {}'.format(new_build_number))
    with open("SUBMISSION_NUMBER", "w") as f:
        f.write(str(new_build_number))

def _create_evaluation_file(test_ids, sub_preds_abs,next_submission_number, increment=True):
    if increment:
        _increment_submission_number(next_submission_number)

    df_submission = pd.DataFrame(
        [test_ids.values, sub_preds_abs]).transpose()
    df_submission.columns = ['OBS_ID', 'success']
    df_submission['OBS_ID'] = df_submission.OBS_ID.astype(int)
    df_submission['success'] = df_submission.success.astype(int)
    fileName = 'data/submissions/submission' + next_submission_number + '.csv'
    df_submission.to_csv(fileName, index=None)
    logger.info("Write submission file to: {}".format(fileName))

def _write_results(feature_set_number, mean_mcc,model_name,next_submission_number, hyperparam={"n_estimators":"1", "max_leafs":"5"}):
    feature_sets = read_feature_meta()
    if (feature_set_number in feature_sets):
        feature_set = feature_sets[feature_set_number]
    else: 
        feature_sets = read_feature_meta(True)
        feature_set = feature_sets[feature_set_number]
    
    result = _read_result_json()

    metrics ={}
    metrics['mcc'] = mean_mcc

    result.append({
        'timestamp': str(time()),
        'feature_set': feature_set_number,
        'features':feature_set,
        'metrics':metrics,
        'model_name':model_name,
        'hyperparam':hyperparam,
        'submission':next_submission_number
    })
    _write_result_json(result)


@click.command()
@click.argument('feature_set')
def main(feature_set):
    logger.info("Building model with featuer set {}".format(feature_set))
    model = LightGbmModel(feature_set)

    next_submission_number = _get_submission_number()

    mean_mcc, hyperparam = model.cross_validation()
    test_ids, sub_preds_abs = model.get_values()
    model_name = model.get_name()

    _create_evaluation_file(test_ids, sub_preds_abs,next_submission_number, True)
    
    _write_results(feature_set, mean_mcc,model_name, next_submission_number, hyperparam)


if __name__ == "__main__":
    main()
