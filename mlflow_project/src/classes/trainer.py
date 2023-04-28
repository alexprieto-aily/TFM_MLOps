from classes.intermediate_step import IntermediateStep

from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

import numpy as np
import pandas as pd
import os

class Trainer(IntermediateStep):
    def __init__(
            self
            , name
            , data_path
            , date_cols
            , target_variable
            , destination_directory
            , random_state
    ):

        super().__init__(name
                         , data_path
                         , date_cols
                         , target_variable
                         , destination_directory)
        

def random_grid_search(model, param_distributions, X_train, y_train, seed):
    """
    Perform a random grid search using scikit-learn's RandomizedSearchCV class.

    Parameters:
    - model: a scikit-learn model object
    - param_distributions: a dictionary containing hyperparameter names and distributions
    - X_train: training input data
    - y_train: training output data
    - n_iter: number of parameter settings that are sampled
    - cv: number of cross-validation folds

    Returns:
    - best_estimator: the best estimator found during the search
    - best_params: the hyperparameters of the best estimator
    """
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=seed)
    # Create a RandomizedSearchCV object

    search = RandomizedSearchCV(
        model, param_distributions, cv=cv, scoring='recall', n_iter=10)

    # Fit the RandomizedSearchCV object to the data
    search.fit(X_train, y_train)

    # Print the best parameters and score
    print("Best parameters: {}".format(search.best_params_))
    print("Best cross-validation score: {:.2f}".format(search.best_score_))

    # Get the best estimator and its parameters
    best_estimator = search.best_estimator_
    best_params = search.best_params_

    return best_estimator, best_params

def cross_val_train_predict(model, X_train, y_train, X_test, scoring_metric, seed=SEED):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=seed)
    scores = cross_val_score(model, X_train, y_train,
                             scoring=scoring_metric, cv=cv, n_jobs=-1)
    print('Mean Recall in Train: %.3f (%.3f)' %
          (np.mean(scores), np.std(scores)))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred