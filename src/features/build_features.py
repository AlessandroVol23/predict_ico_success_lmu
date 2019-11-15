import numpy as np
import click
from src.features.feature_class import FeatureEngineering
from src.features.utils import read_processed_data


@click.command()
def main():
    df_bitcoin, df, df_test = read_processed_data()
    fe = FeatureEngineering(df, df_bitcoin, df_test)
    fe.construct_features()
    X_train, y_train, X_test = fe.get_X_y()
    print("UNIQUE: {}".format(np.unique(y_train, return_counts=True)))
    X_train.to_csv('data/features/features_x_train.csv', index=None)
    y_train.tofile('data/features/features_y_train.np')
    X_test.to_csv('data/features/features_x_test.csv', index=None)


if __name__ == "__main__":
    main()
