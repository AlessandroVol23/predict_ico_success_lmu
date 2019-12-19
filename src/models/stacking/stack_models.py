from src.models.build_model import BuildModel
import click

from src.models.catboost_model import CatBoostModel
from src.models.fit_model import FittingModel
from src.models.light_gbm import LightGbmModel
from src.models.utils import read_upsampling_feature_set, read_categorical_features
from src.utils import read_feature_meta
import pandas as pd

training_models = [
    CatBoostModel,
    LightGbmModel
]


@click.command()
@click.argument('feature_set_key')
def stack_models(feature_set_key):
    feature_set_meta = read_feature_meta()

    df_oof_train = pd.DataFrame()
    df_oof_test = pd.DataFrame()

    for current_model_class in training_models:
        current_model = current_model_class()

        upsampling = read_upsampling_feature_set(
            feature_set_meta, feature_set_key)

        categorical_features = read_categorical_features(
            feature_set_meta, feature_set_key)
        # Fitting model that trains and cross validates, takes the underlying model to train as a param
        fitting_model = FittingModel(
            feature_set_key, current_model, categorical_features, upsample=upsampling)

        # Get values from fitting model
        mean_mcc = fitting_model.cross_validation()
        oof_train, oof_test = fitting_model.get_oof()

        df_oof_train[current_model.get_name()] = oof_train
        df_oof_test[current_model.get_name()] = oof_test

    hyperparam = {
        'iterations': 800,
        'loss_function': 'Logloss',
        # 'loss_function': 'CrossEntropy',
        #  'learning_rate': 0.1,
        'logging_level': 'Verbose'
    }

    clf = CatBoostModel(hyperparam)
    #X_train = df_oof_train.values
    #y_train = df_oof_test.values
    clf.fit(df_oof_train, fitting_model.y_train)
    print("Stacked model trained!")


if __name__ == "__main__":
    stack_models()
