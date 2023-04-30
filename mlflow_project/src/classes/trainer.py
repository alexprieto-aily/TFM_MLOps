from classes.intermediate_step import IntermediateStep

import mlflow

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

import numpy as np


class Trainer(IntermediateStep):
    def __init__(
            self
            , name
            , data_path
            , date_cols
            , target_variable
            , destination_directory
            , model
            , random_state
            , splitter
    ):

        super().__init__(name
                         , data_path
                         , date_cols
                         , target_variable
                         , destination_directory)
        self.model = model
        self.random_state = random_state
        self.splitter = splitter
        self.params = None

        
def random_grid_search(self, param_distributions, seed):
    """
    Perform a random grid search using scikit-learn's RandomizedSearchCV class.

    Parameters:
    - self.model: a scikit-learn self.model object
    - param_distributions: a dictionary containing hyperparameter names and distributions
    - Self.X_train: training input data
    - self.y_train: training output data
    - n_iter: number of parameter settings that are sampled
    - cv: number of cross-validation folds

    Returns:
    - best_estimator: the best estimator found during the search
    - best_params: the hyperparameters of the best estimator
    """
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=self.random_state)
    # Create a RandomizedSearchCV object

    search = RandomizedSearchCV(
        self.model, param_distributions, cv=cv, scoring='recall', n_iter=10)

    # Fit the RandomizedSearchCV object to the data
    search.fit(self.X_train, self.y_train)

    # Print the best parameters and score
    print("Best parameters: {}".format(search.best_params_))
    print("Best cross-validation score: {:.2f}".format(search.best_score_))

    return  search.best_params_




def cross_val_train_predict(self, model, X_train, y_train, X_test, scoring_metric, seed=SEED):
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=seed)
    scores = cross_val_score(model, X_train, y_train,
                             scoring=scoring_metric, cv=cv, n_jobs=-1)
    print('Mean Recall in Train: %.3f (%.3f)' %
          (np.mean(scores), np.std(scores)))
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return y_pred

def set_experiment(self, experiment_name, tracking_uri='http://localhost:5000'):
    # Set the experiment name and tracking URI
    mlflow.set_experiment(experiment_name)
    mlflow.set_tracking_uri(tracking_uri)
    client = mlflow.tracking.MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)
    run = client.create_run(experiment.experiment_id)
    print(f"Experiment run_id={run.info.run_id} created in tracking URI={tracking_uri}")


def run_experiment(self, experiment_name, model_class, params, splitter, log_model=False):
    
    set_experiment(experiment_name)
    mlflow.sklearn.autolog(log_models=log_model)
    # Log the evaluation metrics and model parameters in MLflow
    with mlflow.start_run():
        model = model_class(**params)  # create a new instance of the model class
        model.fit(splitter.X_train, splitter.y_train)
        y_pred = model.predict(splitter.X_test)
        
    return model, y_pred

def delete_experiment(experiment_name):
    client = mlflow.tracking.MlflowClient()
    experiment = mlflow.get_experiment_by_name(experiment_name)

    # Checks if the experiment exists
    if experiment is None:
        print(f"Experiment '{experiment_name}' does not exist.")
        return

    experiment = mlflow.get_experiment_by_name(experiment_name)
    client.delete_experiment(experiment.experiment_id)
           