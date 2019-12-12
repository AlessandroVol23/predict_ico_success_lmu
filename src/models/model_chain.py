import click
from src.utils import read_feature_meta
from src.models.build_model import BuildModel


@click.command()
@click.argument('feature_set')
def main(feature_set):
    feature_set_meta = read_feature_meta()
    keys = feature_set_meta.keys()

    for key in keys:
        print("FOR KEY: {}".format(key))
        model = BuildModel()
        model.train_model(key)


if __name__ == "__main__":
    main()
