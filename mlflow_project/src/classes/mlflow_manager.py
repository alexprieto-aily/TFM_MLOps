import mlflow

class MLflowManager:
    def __init__(self
                 , name
                 ):
        self.name = name
        self.client = mlflow.tracking.MlflowClient()
   


    def set_experiment_mlflow(self, experiment_name, tracking_uri='http://localhost:5000'):
        # Set the experiment name and tracking URI
        mlflow.set_experiment(experiment_name)
        mlflow.set_tracking_uri(tracking_uri)
        
        print(f"Experiment {experiment_name} created in tracking URI {tracking_uri}")

    def log_model_mlflow(self, trainer, infer_signature): 

        if infer_signature:
            signature = mlflow.models.signature.infer_signature(trainer.splitter.X_train, trainer.splitter.y_train)
            mlflow.sklearn.log_model(trainer.model_class, trainer.name, signature=signature)
        else:
            mlflow.sklearn.log_model(trainer.model_class, trainer.name)
        
    def run_mlflow(self, experiment_name, run_name, trainer, log_models=False):
        client = mlflow.tracking.MlflowClient()
        experiment = mlflow.get_experiment_by_name(experiment_name)

        with mlflow.start_run(run_name=run_name, experiment_id=experiment.experiment_id):
           mlflow.log_param("model_name", trainer.name)
           mlflow.log_param("model_params", trainer.model_class.get_params())

        for key, value in trainer.results.items():
            mlflow.log_metric('test_' + key, value)
            if log_models:
                # Infering signature no make sure of correct input and output
                self.log_model_mlflow(infer_signature=True)


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
            