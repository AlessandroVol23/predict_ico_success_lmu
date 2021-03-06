import numpy as np
import click
from src.features.feature_class import FeatureEngineering
from src.features.utils import read_processed_data
import logging
import json
from tqdm import tqdm

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


def _read_feature_meta():
    with open("data/features/feature_set_meta/feature_set_meta.json") as f:
        return (json.load(f))


def _read_random_feature_meta():
    with open("data/features/feature_set_meta/random_feature_sets_meta.json") as f:
        return (json.load(f))


@click.command()
@click.argument('random')
def main(random):
    df_bitcoin, df, df_test, df_gem_btc_usd, df_gem_eth_usd, df_gem_ltc_usd, df_icobench = read_processed_data()
    feature_sets = []
    if random == "True":
        feature_sets = _read_random_feature_meta()
    else:
        feature_sets = _read_feature_meta()

    for (key, value) in tqdm(feature_sets.items()):
        tqdm.write("building features for feature_set {} ".format(key))
        fe = FeatureEngineering(df, df_bitcoin, df_test,
                                df_gem_btc_usd, df_gem_eth_usd, df_gem_ltc_usd, df_icobench)
        fe.construct_feature_set(value)
        X_train, y_train, X_test = fe.get_X_y()
        logger.debug("UNIQUE: {}".format(
            np.unique(y_train, return_counts=True)))
        X_train.to_csv('data/features/feature_sets/features_x_train_' +
                       key + '.csv', index=None)
        y_train.tofile(
            'data/features/feature_sets/features_y_train_' + key + '.np')
        X_test.to_csv(
            'data/features/feature_sets/features_x_test_' + key + '.csv', index=None)


if __name__ == "__main__":
    main()
