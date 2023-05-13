from classes.trainer import Trainer

import mlflow

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn import metrics

import numpy as np


class ClassifierTrainer(Trainer):
    def __init__(
            self
            , name
            , model_class
            , random_state
            , splitter
            , objective_metric
    ):

        super().__init__(name
                         , model_class
                         , random_state
                         , splitter
                         , objective_metric)

       
    
    def train(self):
        print(f"X_train shape: {self.splitter.X_train.shape}")
        self.model_class.fit(self.splitter.X_train, self.splitter.y_train)
        print(f"Model {self.name} trained")

    def train_grid_search(self, param_distributions, n_splits=5, n_repeats=5, n_jobs=-1):
   
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
        search.fit(self.splitter.X_train, self.splitter.y_train,)
        super().set_model_params(search.best_params_)
        self.model_class = search.best_estimator_
        # Print the best parameters and score
        print("Best parameters: {}".format(search.best_params_))
        print("Best cross-validation score: {:.2f}".format(search.best_score_))
        
    
    def predict(self, X_test):
        self.y_pred = self.model_class.predict(X_test)
        print(f"Model {self.name} has made the predictions")

    @staticmethod
    def get_metrics(y_test, y_pred):
        """This function returns a dictionary with the metrics
        """
        results = {}
        results['accuracy'] = metrics.accuracy_score(y_test, y_pred)
        results['precision'] = metrics.precision_score(y_test, y_pred)
        results['recall'] = metrics.recall_score(y_test, y_pred)
        results['f1'] = metrics.f1_score(y_test, y_pred)
        results['roc_auc'] = metrics.roc_auc_score(y_test, y_pred)
        return results

    def evaluate(self, y_test):
        self.results =  self.get_metrics(y_test, self.y_pred)

 
        
    def execute(self):
        pass