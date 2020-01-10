# Predicting ICO Success 

This repository shows our internal Kaggle challenge of the module "Data Science for Business" of the IBIS institute at LMU Munich.

## Environment

To install all dependencies you need pipenv. After installing pipenv just type:

`pipenv shell`

`pipenv install`

After that you have all dependencies and you are ready to go. For further instructions please visit https://github.com/pypa/pipenv.

## Data

Preprocess data with `make data`

- Basic preprocessing steps

## Features

`make features`:
- Generate all features which were declared in config meta file: `data/features/feature_set_meta/feature_set_meta.json`

Example feature set:
```json
{
  "feature_set_1": [
    {
      "meta": {
        "upsampling": "0.0"
      }
    },
    {
      "column": "transaction_count",
      "type": "numerical",
      "na_strategy": "median" 
    }
    {
      "column": "timestamp",
      "type": "binary",
      "na_strategy": "False"
    },
    {
      "column": "divided_total_supplie_transcation_count",
      "type": "divide", 
      "na_strategy": "set:1",
      "columns": ["market_data_total_supply", "transaction_count"]
    }
  ]
}
```

- column: Which column of the data to take?
- Type: Which type of data? [numerical, categorical, binary, divide, duration, multiply, add, difference, average, coefficient]
- na_strategy: How to handle NAs? [mean, median, set:x, False]
- encoder: Encoder for categorical features: [label, one_hot]


## Models

Different types of models were implemented. Following models were implemented:

1. Random Forest
2. LightGbm
3. Catboost
4. Naive Bayes
5. Logistic Regression

`make train_models feature_set=feature_set`: All models will be trained on `feature_set`. After training the results of the training will be saved in a separate results.json in `data/results/results.json`.

`make model_chain`: Trains all models on all feature sets, declared in `feature_set_meta.json`.

### Stacking

A stacking approach was implemented as well. With `make stacking feature_set=feature_set` all models will be validated (5-fold), trained on the whole dataset and a submission file will be created afterwards.

### Cross-Validation

The datasets cross validates on 5-folds, trains on the whole dataset and creates submission files for both cases. 

## Further Make commands


### Random Feature Sets

`make features_random_meta`:
- Creates random feature sets without custom features

### shap_plots

`make shap_plots`: Trains all models on all feature sets declared in `feature_set_meta.json` and creates shap plots for every model and feature set in `reports/figures/local`. 

## End Results

In the end we could make first place with a MCC on the private leaderboard of 0.42 MCC whereas the public score 0.48 MCC was.
