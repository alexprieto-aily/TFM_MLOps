from classes.trainer import Trainer

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import numpy as np
import pandas as pd

class ClassifierTrainer(Trainer):
    def __init__(
        self,
        name: str,
        model_class,
        random_state: int,
        splitter,
        objective_metric: str
    ):
        """
        Initializes a ClassifierTrainer object.

        Args:
            name (str): The name of the trainer.
            model_class: The classifier model class to use.
            random_state (int): The random seed for reproducibility.
            splitter: The data splitter object for train-test split.
            objective_metric (str): The evaluation metric to optimize during training.
        """
        super().__init__(name
                         , model_class
                         , random_state
                         , splitter
                         , objective_metric)

    def train(self) -> None:
        """
        Trains the classifier model using the training data.
        """
        print(f"X_train shape: {self.splitter.X_train.shape}")
        self.model_class.fit(self.splitter.X_train, self.splitter.y_train)
        print(f"Model {self.name} trained")

    def train_grid_search(
        self,
        param_distributions,
        n_splits: int = 5,
        n_repeats: int = 5,
        n_jobs: int = -1
    ) -> None:
        """
        Performs grid search with cross-validation to find the best hyperparameters
        for the classifier model.

        Args:
            param_distributions: The parameter distributions to search over.
            n_splits (int): The number of splits in the cross-validation.
            n_repeats (int): The number of times cross-validation is repeated.
            n_jobs (int): The number of parallel jobs to run (-1 means using all available processors).

        """
        
        cv = RepeatedStratifiedKFold(n_splits=n_splits
                                    , n_repeats=n_repeats
                                    , random_state=self.random_state
                                    )
        search = GridSearchCV(
            self.model_class
            , param_distributions
            , cv=cv
            , scoring=self.objective_metric
            , n_jobs=n_jobs)

        print(f"Fitting grid search with {n_splits} splits and {n_repeats} repeats")
        print(f"X_train shape: {self.splitter.X_train.shape}")
        search.fit(self.splitter.X_train, self.splitter.y_train)
        super().set_model_params(search.best_params_)
        self.model_class = search.best_estimator_
        # Print the best parameters and score
        print("Best parameters: {}".format(search.best_params_))
        print("Best cross-validation score: {:.2f}".format(search.best_score_))

    def predict(self, X_test: pd.DataFrame ) -> None:
        """
        Makes predictions using the trained classifier model.

        Args:
            X_test: The test data to make predictions on.
        """
        self.y_pred = self.model_class.predict(X_test)
        print(f"Model {self.name} has made the predictions")

    @staticmethod
    def get_metrics(y_test, y_pred: pd.DataFrame) -> dict:
        """
        Computes evaluation metrics for the classifier predictions.

        Args:
            y_test: The true labels.
            y_pred: The predicted labels.

        Returns:
            dict: A dictionary containing the computed metrics.
        """
        results = {}
        results['accuracy'] = metrics.accuracy_score(y_test, y_pred)
        results['precision'] = metrics.precision_score(y_test, y_pred)
        results['recall'] = metrics.recall_score(y_test, y_pred)
        results['f1'] = metrics.f1_score(y_test, y_pred)
        results['roc_auc'] = metrics.roc_auc_score(y_test, y_pred)
        return results

    def evaluate(self, y_test: pd.Series) -> None:
        """
        Evaluates the classifier predictions using the test data.
        """
        self.results =  self.get_metrics(y_test, self.y_pred)

    def execute(self):
        pass
