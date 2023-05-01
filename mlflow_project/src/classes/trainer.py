from classes.step import Step

import mlflow

from sklearn.model_selection import train_test_split
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn import metrics

import numpy as np


class Trainer(Step):
    def __init__(
            self
            , name
            , model_class
            , random_state
            , splitter
    ):

        super().__init__(name)
        self.params = None
        
        if self.params:
            self.model_class = model_class(**self.params)
        else:
            self.model_class = model_class

        self.random_state = random_state
        self.splitter = splitter
        self.y_pred = None
        
 
            
    def random_grid_search(self, param_distributions, scoring_metric,  n_splits=10, n_repeats=3):
   
        cv = RepeatedStratifiedKFold(n_splits=n_splits
                                    , n_repeats=n_repeats
                                    , random_state=self.random_state
                                    )
        # Create a RandomizedSearchCV object

        search = RandomizedSearchCV(
            self.model_class, param_distributions, cv=cv, scoring=scoring_metric, n_iter=10)

        # Fit the RandomizedSearchCV object to the data
        search.fit(self.splitter.X_train, self.splitter.y_train)

        # Print the best parameters and score
        print("Best parameters: {}".format(search.best_params_))
        print("Best cross-validation score: {:.2f}".format(search.best_score_))

        return  search.best_params_

    def set_model_params(self, params):
        self.params = params
        self.model_class = self.model_class.set_params(**self.params)


    def cross_val_train_predict(self,  scoring_metric, n_splits=10, n_repeats=3):
        cv = RepeatedStratifiedKFold(n_splits=n_splits
                                    , n_repeats=n_repeats
                                    , random_state=self.random_state
                                    )
        
        scores = cross_val_score(self.model_class
                                , self.splitter.X_train
                                , self.splitter.y_train,
                                scoring=scoring_metric
                                , cv=cv, n_jobs=-1
                                )
        print('Mean {scoring_metric} in Train: %.3f (%.3f)' %
            (np.mean(scores), np.std(scores)))
        self.model_class.fit(self.splitter.X_train, self.splitter.y_train)
        self.y_pred = self.model_class.predict(self.splitter.X_test)
  

    def set_experiment_mlflow(self, experiment_name, tracking_uri='http://localhost:5000'):
        # Set the experiment name and tracking URI
        mlflow.set_experiment(experiment_name)
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        experiment = mlflow.get_experiment_by_name(experiment_name)
        run = client.create_run(experiment.experiment_id)
        print(f"Experiment run_id={run.info.run_id} created in tracking URI={tracking_uri}")


    def run_experiment_mlflow(self, experiment_name, log_models=False):
        
        self.set_experiment_mlflow(experiment_name)
        mlflow.sklearn.autolog(log_models=log_models)
        # Log the evaluation metrics and self.model_class parameters in MLflow
        with mlflow.start_run():
            # self.model_class = self.model_class_class(**params)  # create a new instance of the self.model_class class
            self.model_class.fit(self.splitter.X_train, self.splitter.y_train)
            self.y_pred = self.model_class.predict(self.splitter.X_test)
            accuracy = metrics.accuracy_score(self.splitter.y_test, self.y_pred)
            precision = metrics.precision_score(self.splitter.y_test, self.y_pred)
            recall = metrics.recall_score(self.splitter.y_test, self.y_pred)
            f1_score = metrics.f1_score(self.splitter.y_test, self.y_pred)
            roc_auc = metrics.roc_auc_score(self.splitter.y_test, self.y_pred)

            mlflow.log_metric("test_accuracy", accuracy)
            mlflow.log_metric("test_precision", precision)
            mlflow.log_metric("test_recall", recall)
            mlflow.log_metric("test_f1_score", f1_score)
            mlflow.log_metric("test_roc_auc", roc_auc)
            

    def delete_experiment_mlflow(self, experiment_name):
        client = mlflow.tracking.MlflowClient()
        experiment = mlflow.get_experiment_by_name(experiment_name)

        # Checks if the experiment exists
        if experiment is None:
            print(f"Experiment '{experiment_name}' does not exist.")
            return

        experiment = mlflow.get_experiment_by_name(experiment_name)
        client.delete_experiment(experiment.experiment_id)
            
    def execute(self):
        pass
