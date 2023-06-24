import mlflow

class MLflowManager:
    def __init__(self
                 , name: str
                 ):
        """
        Initializes the MLflowManager.

        Args:
            name (str): The name of the MLflowManager.
        """
        self.name = name
        self.tracking_uri = 'http://127.0.0.1:5000'

    @staticmethod
    def set_experiment(experiment_name: str, tracking_uri: str):
        """
        Sets the experiment name and tracking URI in MLflow.

        Args:
            experiment_name (str): The name of the experiment.
            tracking_uri (str): The URI of the MLflow tracking server.
        """
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)

    @staticmethod
    def log_model(trainer, infer_signature: bool):
        """
        Logs the trained model in MLflow.

        Args:
            trainer: The trainer object.
            infer_signature (bool): Whether to infer the model signature or not.
        """
        if infer_signature:
            signature = mlflow.models.signature.infer_signature(trainer.splitter.X_train
                                                                , trainer.splitter.y_train)
            mlflow.sklearn.log_model(sk_model=trainer.model_class
                                     , artifact_path="sklearn-model"
                                     , registered_model_name=trainer.name
                                     , signature=signature)
        else:
            mlflow.sklearn.log_model(sk_model=trainer.model_class
                                     , artifact_path="sklearn-model"
                                     , registered_model_name=trainer.name)
        print(f"Model {trainer.name} logged in MLflow")

    def make_run(self, run_name: str, experiment_name: str, trainer, log_model: bool = False):
        """
        Makes an MLflow run for tracking model performance.

        Args:
            run_name (str): The name of the MLflow run.
            experiment_name (str): The name of the experiment.
            trainer: The trainer object.
            log_model (bool): Whether to log the model or not.
        """
        self.set_experiment(experiment_name, self.tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)

        with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id):
                    
           mlflow.log_param("model_name", trainer.name)
           mlflow.log_param("model_params", trainer.model_class.get_params())

           for key, value in trainer.results.items():
               mlflow.log_metric('test_' + key, value)

           if log_model:
                # Infering signature no make sure of correct input and output
                self.log_model(trainer, infer_signature=True)

        mlflow.end_run()

        print(f"Run completed")

    def log_new_metrics(self, run_name: str, experiment_name: str, metrics: dict, prefix: str = ''):
        """
        Logs new metrics for an existing MLflow run.

        Args:
            run_name (str): The name of the MLflow run.
            experiment_name (str): The name of the experiment.
            metrics (dict): The new metrics to log.
            prefix (str): The prefix to add to the metric names.
        """
        self.set_experiment(experiment_name, self.tracking_uri)
        experiment = mlflow.get_experiment_by_name(experiment_name)
        runs = mlflow.search_runs(experiment_ids=mlflow.get_experiment_by_name(experiment_name).experiment_id)
        run = runs.loc[runs['tags.mlflow.runName'] == run_name]   

        if len(run) > 0:
            run_id = run['run_id'].values[0]
        else:
            print(f"Run {run_name} does not exist")
            return

        with mlflow.start_run(run_id=run_id, experiment_id=experiment.experiment_id):
           for key, value in metrics.items():
               mlflow.log_metric(prefix + key, value)

        mlflow.end_run()

        print(f"{run_name}")

    @staticmethod
    def delete_experiment(experiment_name: str):
        """
        Deletes an MLflow experiment.

        Args:
            experiment_name (str): The name of the experiment.
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)

        if experiment is None:
            print(f"Experiment '{experiment_name}' does not exist.")
        elif experiment.lifecycle_stage == "deleted":
            # Permanently delete the experiment
            mlflow.delete_experiment(experiment.experiment_id)
            print(f"Experiment '{experiment_name}' has been permanently deleted.")
        else:
            print(f"Experiment '{experiment_name}' is not deleted.")
    
    @staticmethod
    def get_experiment_id(experiment_name: str):
        """
        Retrieves the experiment ID for a given experiment name.

        Args:
            experiment_name (str): The name of the experiment.

        Returns:
            str: The experiment ID.
        """
        experiment = mlflow.get_experiment_by_name(experiment_name)
        return experiment.experiment_id

    @staticmethod
    def set_model_stage(model_name: str, model_version: str, stage: str):
        """
        Sets the stage for a registered model.

        Args:
            model_name (str): The name of the registered model.
            model_version (str): The version of the registered model.
            stage (str): The stage to set for the registered model.
        """
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage=stage
        )

    @staticmethod
    def load_model(model_name: str, model_version: str, tracking_uri: str = 'http://127.0.0.1:5000'):
        """
        Loads a registered model from MLflow.

        Args:
            model_name (str): The name of the registered model.
            model_version (str): The version of the registered model.
            tracking_uri (str): The URI of the MLflow tracking server.

        Returns:
            object: The loaded model.
        """
        mlflow.set_tracking_uri(tracking_uri)
        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.sklearn.load_model(model_uri=model_uri)
        print(f"Model {model_name} loaded from MLflow")
        return model

    @staticmethod
    def list_registered_models(tracking_uri: str = 'http://127.0.0.1:5000'):
        """
        Lists all registered models in MLflow.

        Args:
            tracking_uri (str): The URI of the MLflow tracking server.

        Returns:
            list: The list of registered models.
        """
        client = MlflowClient(tracking_uri=tracking_uri)
        models = client.list_registered_models()
        return models

    @staticmethod
    def get_model_path(model_name: str, model_version: str):
        """
        Retrieves the path of a registered model.

        Args:
            model_name (str): The name of the registered model.
            model_version (str): The version of the registered model.

        Returns:
            str: The path of the registered model.
        """
        client = mlflow.tracking.MlflowClient()
        model_version_details = client.get_model_version(model_name, model_version)
        model_path = model_version_details.source

        return model_path