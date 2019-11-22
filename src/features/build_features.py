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


def read_feature_meta():
    with open("data/features/feature_set_meta/feature_set_meta.json") as f:
        return (json.load(f))


@click.command()
def main():
    df_bitcoin, df, df_test = read_processed_data()
    feature_sets = read_feature_meta()

    for(key, value) in tqdm(feature_sets.items()):
        tqdm.write("building features for feature_set {} ".format(key))
        fe = FeatureEngineering(df, df_bitcoin, df_test)
        fe.construct_feature_set(value)
        X_train, y_train, X_test = fe.get_X_y()
        logger.debug("UNIQUE: {}".format(
            np.unique(y_train, return_counts=True)))
        X_train.to_csv('data/features/features_x_train_' +
                       key+'.csv', index=None)
        y_train.tofile('data/features/features_y_train_'+key+'.np')
        X_test.to_csv('data/features/features_x_test_'+key+'.csv', index=None)



if __name__ == "__main__":
    main()
