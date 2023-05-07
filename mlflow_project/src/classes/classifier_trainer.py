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
        search.fit(self.splitter.X_train, self.splitter.y_train,)
        super().set_model_params(search.best_params_)
        self.model_class = search.best_estimator_
        # Print the best parameters and score
        print("Best parameters: {}".format(search.best_params_))
        print("Best cross-validation score: {:.2f}".format(search.best_score_))
        
    
    def predict(self):
        self.y_pred = self.model_class.predict(self.splitter.X_test)
        print(f"Model {self.name} has made the predictions")

    def evaluate(self, y_test, y_pred):
        """This function evaluates the model
        and returns a dictionary with the results
        """ 
        self.predict()
        results = {}
        results['accuracy'] = metrics.accuracy_score(y_test, y_pred)
        results['precision'] = metrics.precision_score(y_test, y_pred)
        results['recall'] = metrics.recall_score(y_test, y_pred)
        results['f1'] = metrics.f1_score(y_test, y_pred)
        results['roc_auc'] = metrics.roc_auc_score(y_test, y_pred)
        return results

 
            
    def set_experiment_mlflow(self, experiment_name, tracking_uri='http://localhost:5000'):
        # Set the experiment name and tracking URI
        mlflow.set_experiment(experiment_name)
        mlflow.set_tracking_uri(tracking_uri)
        client = mlflow.tracking.MlflowClient()
        experiment = mlflow.get_experiment_by_name(experiment_name)
        run = client.create_run(experiment.experiment_id)
        print(f"Experiment run_id={run.info.run_id} created in tracking URI={tracking_uri}")

    def log_model_mlflow(self, infer_signature): 

        if infer_signature:
            signature = mlflow.models.signature.infer_signature(self.splitter.X_train, self.splitter.y_train)
            mlflow.sklearn.log_model(self.model_class, self.name, signature=signature)
        else:
            mlflow.sklearn.log_model(self.model_class, self.name)
        
    def run_mlflow(self, run_tag, log_models=False, infer_signature=True):
        with mlflow.start_run():
            self.evaluate()
            mlflow.log_param("model_name", self.name)
            mlflow.log_param("model_params", self.model_class.get_params())

            for key, value in self.results.items():
                mlflow.log_metric('test_' + key, value)

            if log_models:
                self.log_model_mlflow(infer_signature=infer_signature)


    def run_experiment_mlflow(self, experiment_name, log_models=False, infer_signature=True):
        
        self.set_experiment_mlflow(experiment_name)
        mlflow.sklearn.autolog(log_models=log_models)

        # Log the evaluation metrics and self.model_class parameters in MLflow
        

    @staticmethod
    def delete_experiment_mlflow(experiment_name):
        client = mlflow.tracking.MlflowClient()
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            print(f"Experiment '{experiment_name}' does not exist.")
        elif experiment.lifecycle_stage == "deleted":
            # Permanently delete the experiment
            mlflow.delete_experiment(experiment.experiment_id)
            print(f"Experiment '{experiment_name}' has been permanently deleted.")
        else:
            print(f"Experiment '{experiment_name}' is not deleted.")
            

    def execute(self):
        pass