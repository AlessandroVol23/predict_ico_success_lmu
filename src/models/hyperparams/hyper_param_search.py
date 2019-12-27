from src.models.utils import read_feature_data
from skopt.space import Real, Integer
from skopt import BayesSearchCV
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.model_selection import StratifiedKFold
from src.models.random_forest import RandomForestModel
from src.models.naive_bayes import NaiveBayesModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.light_gbm_dart import LightGbmDartModel
from src.models.light_gbm import LightGbmModel
from src.models.fit_model import FittingModel
from src.models.catboost_model import CatBoostModel
from catboost import CatBoostClassifier
import json
import warnings

import click
from joblib import dump
import logging
import time

warnings.filterwarnings("ignore")

# Classifiers

# Model selection

# Metrics

# Skopt functions

search_models = [

    CatBoostModel,
    LightGbmModel,
    NaiveBayesModel,
    RandomForestModel,
    LogisticRegressionModel,
    LightGbmDartModel
]

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

MAX_ITERATIONS = 1


def _write_result(result, opt):
    with open('reports/best_params.json', 'w') as outfile:
        json.dump(result, outfile)

    dump(opt, 'optimizer.joblib')


@click.command()
@click.argument('feature_set_key')
def main(feature_set_key):
    result = []
    for current_model_class in search_models:
        start = time.time()
        x_train, y_train, x_test = read_feature_data(
            feature_set=feature_set_key)

        if current_model_class == CatBoostModel:
            current_model = current_model_class({"thread_count": 2,
                                                 "loss_function": 'Logloss',
                                                 "od_type": 'Iter',
                                                 "verbose": True})
            search_spaces = {'iterations': Integer(10, 1000),
                             'depth': Integer(1, 8),
                             'learning_rate': Real(0.01, 1.0, 'log-uniform'),
                             'random_strength': Real(1e-9, 10, 'log-uniform'),
                             'bagging_temperature': Real(0.0, 1.0),
                             'border_count': Integer(1, 255),
                             'l2_leaf_reg': Integer(2, 30),
                             'scale_pos_weight': Real(0.01, 1.0, 'uniform')}
        elif current_model_class == LightGbmModel:
            current_model = current_model_class({'num_iterations': 100,
                                                 'learning_rate': 0.05,
                                                 'metric': 'lgb_r2_score'})
            search_spaces = {'num_leaves': Integer(24, 45),
                             'feature_fraction': Real(0.1, 0.9),
                             'bagging_fraction': Real(0.8, 1),
                             'max_depth': Integer(5, 9),
                             'lambda_l1': Integer(0, 5),
                             'lambda_l2': Integer(0, 3),
                             'min_split_gain': Real(0.001, 0.1),
                             'min_child_weight': Integer(5, 50)}
        elif current_model_class == RandomForestModel:
            current_model = current_model_class({"bootstrap": True, "class_weight": None, "criterion": 'gini',
                                                 "max_depth": None, "max_features": 'auto', "max_leaf_nodes": None,
                                                 "min_samples_leaf": 1,
                                                 "min_samples_split": 2, "min_weight_fraction_leaf": 0.0,
                                                 "n_estimators": 10, "n_jobs": 1, "oob_score": False,
                                                 "verbose": 0, "warm_start": False})
            search_spaces = {'criterion': ['gini', 'entropy'], 'max_depth': [3, 4, 6, 10], 'n_estimators': [
                100, 200, 500], 'max_features': ['sqrt', 'log2', 0.2, 0.5, 0.8], 'min_samples_split': [2, 5, 20, 50]}
        elif current_model_class == LogisticRegressionModel:
            current_model = current_model_class(
                {"penalty": 'l1', "dual": False, "max_iter": 110})
            search_spaces = {
                "dual": [True, False],
                "max_iter": [100, 110, 120, 130, 140],
                "C": [1.0, 1.5, 2.0, 2.5]
            }
        elif current_model_class == LightGbmDartModel:
            current_model = current_model_class({'num_iterations': 100,
                                                 'learning_rate': 0.05,
                                                 'metric': 'lgb_r2_score'})
            search_spaces = {'num_leaves': Integer(24, 45),
                             'feature_fraction': Real(0.1, 0.9),
                             'bagging_fraction': Real(0.8, 1),
                             'max_depth': Integer(5, 9),
                             'lambda_l1': Integer(0, 5),
                             'lambda_l2': Integer(0, 3),
                             'min_split_gain': Real(0.001, 0.1),
                             'min_child_weight': Integer(5, 50)}
        else:
            logger.warn("Model not found. Continue")
            continue

        logger.info("Hyperparam search for model {} started".format(
            current_model.get_name()))
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

        scorer = make_scorer(matthews_corrcoef)

        opt = BayesSearchCV(current_model.get_model(), search_spaces, scoring=scorer, cv=skf, n_iter=MAX_ITERATIONS, n_jobs=1, return_train_score=False,
                            refit=True, optimizer_kwargs={'base_estimator': 'GP'}, random_state=42)

        opt.fit(x_train, y_train)
        result.append({
            "model": current_model.get_name(),
            **opt.best_params_
        })
        end = time.time()
        logger.info("Hyperparam search for model {} finished after {}  seconds".format(
            current_model.get_name(), end - start))

    _write_result(result, opt)
    logger.info("Wrote results for all models")


if __name__ == "__main__":
    main()
