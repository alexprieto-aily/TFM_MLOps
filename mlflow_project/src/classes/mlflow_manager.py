import mlflow

class MLflowManager:
    def __init__(self
                 , name
                 ):
        self.name = name
        self.tracking_uri = 'http://localhost:5000'  

    @staticmethod
    def set_experiment(experiment_name, tracking_uri):
        # Set the experiment name and tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        print(f"Experiment {experiment_name} created in tracking URI {tracking_uri}")
    
    @staticmethod
    def log_model( trainer, infer_signature): 
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


    def make_run(self, run_name, experiment_name, trainer, log_model=False):
        
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

    def log_new_metrics(self
                        , run_name
                        , experiment_name
                        , metrics
                        , prefix):
        
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

        print(f"New metrics logged in run {run_name}")

    @staticmethod
    def delete_experiment(experiment_name):

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
    def get_experiment_id(experiment_name):
        experiment = mlflow.get_experiment_by_name(experiment_name)
        return experiment.experiment_id
    
    @staticmethod
    def set_model_production(model_name, model_version):
        client = mlflow.tracking.MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=model_version,
            stage='production'
        )

    @staticmethod
    def load_model(model_name, model_version, tracking_uri='http://localhost:5000'):
        mlflow.set_tracking_uri(tracking_uri)

        model_uri = f"models:/{model_name}/{model_version}"
        model = mlflow.sklearn.load_model(model_uri=model_uri)
        print(f"Model {model_name} loaded from MLflow")
        return model
    
    @staticmethod
    def list_registered_models(tracking_uri='http://localhost:5000'):
        client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
        models = client.list_registered_models()
        return models