from classes.intermediate_step import IntermediateStep
from preprocessing import (
CLEAN_DATA_PATH
, DATA_FOLDER
,true_labels
,date_cols
,target_variable
)

from evidently.pipeline.column_mapping import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
import mlflow


class DataLoader(IntermediateStep):
    def __init__(
        self,
        name: str,
        data_path: str,
        date_cols: list,
        true_labels: list,
        target_variable: str,
        destination_directory: str = None
    ):
        """
        Initializes the FeatureEngineer.

        Args:
            name (str): The name of the feature engineer.
            data_path (str): The path to the data.
            date_cols (list): List of date columns in the data.
            true_labels (list): List of true labels for the target variable.
            target_variable (str): The target variable column name.
            destination_directory (str): The directory to save the processed data.
        """
        super().__init__(
            name,
            data_path,
            date_cols,
            target_variable,
            destination_directory
        )
        self.true_labels = true_labels

    def execute(self):
        """
        Executes the feature engineering process.
        """
        print(f"Executing {self.name} step")
        self.load_data(self.data_path, self.date_cols, index_col=0)
        self.data = self.data[self.data.issue_d.dt.year>=2015]
        print(f"Finished executing {self.name} step")


def metrics_list(report):

    metrics = []

    for feature in reference.columns:
        drifts.append((feature, report["metrics"][1]["result"]["drift_by_columns"][feature]["drift_score"]))

    return metrics


if __name__ == "__main__":

    data_loader = DataLoader(
        name = "Data loader"
        , data_path = CLEAN_DATA_PATH
        , date_cols = date_cols
        , true_labels = true_labels
        , target_variable = target_variable
        , destination_directory = DATA_FOLDER
    )
    data_loader.execute()



    data_columns = ColumnMapping()
    data_columns.categorical_features = data_loader.data.select_dtypes(include='object').columns
    data_columns.numerical_features = data_loader.data.select_dtypes(include='number').columns

    mlflow.set_experiment('Data Drift Evaluation with Evidently')

    # Loop over each year present in the 'issue_d' column
    for year in data_loader.data['issue_d'].dt.year.unique():
        # Filter the data for the current year
        previous_year_data = data_loader.data[data_loader.data['issue_d'].dt.year == year - 1]
        current_year_data = data_loader.data[data_loader.data['issue_d'].dt.year == year]

        # Creating the drift report
        if not previous_year_data.empty:
            drift_report = Report(metrics=[DataDriftPreset()])
            drift_report.run(reference_data=previous_year_data, current_data=current_year_data, column_mapping=data_columns)

            with mlflow.start_run(run_name=f"Drift report {year}"):
                mlflow.log_artifact(drift_report.get_html())
                for metric, value in metrics_list(drift_report):
                    mlflow.log_metric(f"drift_{metric}", value)

        # Creating the quality report
        quality_report = Report(metrics=[DataQualityPreset()])
        quality_report.run(current_data=current_year_data, reference_data = None, column_mapping=None)

        with mlflow.start_run(run_name=f"Quality reports {year}"):
            mlflow.log_artifact(quality_report.get_html())
            for metric, value in metrics_list(quality_report):
                mlflow.log_metric(f"quality_{metric}", value)
