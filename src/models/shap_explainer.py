from src.models.catboost_model import CatBoostModel
from src.models.fit_model import FittingModel
from src.models.utils import read_feature_data
from src.utils import read_feature_meta
from src.models.build_model import read_upsampling_feature_set
import click
import shap
import matplotlib.pyplot as plt


def evaluate_model(feature_set_key):
    model = CatBoostModel({
                'iterations': 500,
                'loss_function': 'Logloss',
                'early_stopping_rounds': 300,
                'logging_level': 'Verbose'
            })
    feature_set_meta = read_feature_meta()
    X, y, X_test = read_feature_data(
        feature_set_key)

    upsampling = read_upsampling_feature_set(
        feature_set_meta, feature_set_key)

    # load JS visualization code to notebook
    shap.initjs()

    model.fit(X, y)

    # explain the model's predictions using SHAP
    explainer = shap.TreeExplainer(model.get_model(reinitialize=False))
    shap_values = explainer.shap_values(X)

    # visualize the first prediction's explanation (use matplotlib=True to avoid Javascript)
    shap.summary_plot(shap_values, X, plot_type='bar', show=False)
    plt.savefig('shap_plot.png', bbox_inches='tight')

@click.command()
@click.argument('feature_set_key')
def main(feature_set_key,):
    evaluate_model(feature_set_key)


if __name__ == "__main__":
    main()
