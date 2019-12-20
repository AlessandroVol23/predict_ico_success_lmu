import click
from src.utils import read_feature_meta
from src.models.build_model import BuildModel


@click.command()
@click.argument('feature_set')
@click.argument('final_model')
def main(feature_set, final_model):
    if final_model == 'True':
        final_model = True
    else:
        final_model = False

    feature_set_meta = read_feature_meta()
    keys = feature_set_meta.keys()

    for key in keys:
        model = BuildModel()
        model.train_model(key, final_model=final_model)


if __name__ == "__main__":
    main()
