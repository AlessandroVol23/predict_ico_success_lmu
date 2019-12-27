import json
import warnings

import click
from joblib import dump

warnings.filterwarnings("ignore")

# Classifiers
from catboost import CatBoostClassifier

# Model selection
from sklearn.model_selection import StratifiedKFold

# Metrics
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import make_scorer

# Skopt functions
from skopt import BayesSearchCV
from skopt.space import Real, Integer
from src.models.utils import read_feature_data


@click.command()
@click.argument('feature_set_key')
def main(feature_set_key):
    x_train, y_train, x_test = read_feature_data(feature_set=feature_set_key)

    search_spaces = {'iterations': Integer(10, 1000),
                     'depth': Integer(1, 8),
                     'learning_rate': Real(0.01, 1.0, 'log-uniform'),
                     'random_strength': Real(1e-9, 10, 'log-uniform'),
                     'bagging_temperature': Real(0.0, 1.0),
                     'border_count': Integer(1, 255),
                     'l2_leaf_reg': Integer(2, 30),
                     'scale_pos_weight': Real(0.01, 1.0, 'uniform')}

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)

    clf = CatBoostClassifier(thread_count=2,
                             loss_function='Logloss',
                             od_type='Iter',
                             verbose=True
                             )

    scorer = make_scorer(matthews_corrcoef)

    opt = BayesSearchCV(clf, search_spaces, scoring=scorer, cv=skf, n_iter=1, n_jobs=1, return_train_score=False,
                        refit=True, optimizer_kwargs={'base_estimator': 'GP'}, random_state=42)



    opt.fit(x_train, y_train)

    print(json.dumps(opt.best_params_, indent=4))

    result = {
        "model":"catboost",
        **opt.best_params_
    }

    with open('best_params.json', 'w') as outfile:
        json.dump(result, outfile)

    dump(opt, 'optimizer.joblib')





if __name__ == "__main__":
    main()
