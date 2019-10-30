#! /bin/bash
# -*- coding: utf-8 -*-

import logging
import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)
log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)


def create_submission_csv(model, X_test, path_to_save_csv):
    """Function to create submission csv to upload on Kaggle.

    Parameters
    ----------
    model : model
        Pre-trained model
    X_test : DataFrame
        X_test DataFrame withou labels. Not the validation set!
    path_to_save_csv : str
        String where to save csv file.
    """
    # Get prediction probabilities for test set
    preds = model.predict(
        X_test, num_iteration=model.best_iteration)

    # Probabilities to classes
    classes_preds = preds.round().astype(int)
    df_submission = pd.DataFrame(classes_preds)
