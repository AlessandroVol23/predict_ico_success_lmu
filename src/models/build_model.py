import logging

import click

from src.models.catboost_model import CatBoostModel
from src.models.fit_model import FittingModel
from src.models.light_gbm import LightGbmModel
from src.models.light_gbm_dart import LightGbmDartModel
from src.models.logistic_regression import LogisticRegressionModel
from src.models.naive_bayes import NaiveBayesModel
from src.models.random_forest import RandomForestModel
from src.models.utils import read_upsampling_feature_set, read_categorical_features, get_submission_number, \
    create_evaluation_file, write_results
from src.utils import read_feature_meta

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)

training_models = [
    CatBoostModel,
    LightGbmModel,
    NaiveBayesModel,
    RandomForestModel,
    LogisticRegressionModel,
    LightGbmDartModel
]


class BuildModel(object):

    def __init__(self):
        pass

    def train_model(self, feature_set_key, modelName="", final_model=True):
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

            categorical_features = read_categorical_features(
                feature_set_meta, feature_set_key)
            # Fitting model that trains and cross validates, takes the underlying model to train as a param
            fitting_model = FittingModel(
                feature_set_key, current_model, categorical_features, upsample=upsampling)

            # Get values from fitting model
            mean_mcc = fitting_model.cross_validation()

            test_ids, sub_preds_abs = fitting_model.get_values()

            next_submission_number = get_submission_number()
            create_evaluation_file(test_ids, sub_preds_abs,
                                   next_submission_number, True)

            # get name and params from underlying model
            model_name = current_model.get_name()
            hyperparams = current_model.get_params()

            write_results(feature_set_meta, feature_set_key, mean_mcc, model_name,
                          next_submission_number, hyperparams)

            if final_model:
                # Workaround till we have function to read in optimized hyperparams
                if model_name == 'catboost':
                    hp = {"bagging_temperature": 1.0, "border_count": 202, "depth": 5, "iterations": 705,
                          "l2_leaf_reg": 30, "learning_rate": 0.29502864152955893, "random_strength": 10.0,
                          "scale_pos_weight": 1.0}
                    final_model = current_model_class(hp)
                else:
                    final_model = current_model_class()

                fitting_model = FittingModel(
                    feature_set_key, final_model, categorical_features, upsample=upsampling)

                logger.info("Create final model.")
                fitting_model.train_final_model()
                preds_test = fitting_model.predict_test_set()
                preds_test_abs = preds_test.argmax(axis=1)
                fitting_model.save_current_model()
                if model_name == 'catboost':
                    fitting_model.save_feature_importance('summary')
                    fitting_model.save_feature_importance('shap')
                    fitting_model.save_feature_importance('feature_importance')
                create_evaluation_file(fitting_model.test_ids, preds_test_abs,
                                       next_submission_number, True, '_final')


@click.command()
@click.argument('feature_set_key')
@click.argument('model_name')
def main(feature_set_key, model_name):
    build_model = BuildModel()
    build_model.train_model(feature_set_key, model_name)


if __name__ == "__main__":
    main()
