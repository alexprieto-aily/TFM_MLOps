import pandas as pd
import os
from classes.step import Step
from abc import abstractmethod

class IntermediateStep(Step):
    def __init__(
        self,
        name: str,
        data_path: str,
        date_cols: list,
        target_variable: str,
        destination_directory: str
    ):
        """
        Initializes the IntermediateStep.

        Args:
            name (str): The name of the intermediate step.
            data_path (str): The path to the data file.
            date_cols (list): The list of columns to parse as dates.
            target_variable (str): The name of the target variable.
            destination_directory (str): The directory to save the processed data.
        """
        super().__init__(name)
        self.data = None
        self.data_path = data_path
        self.date_cols = date_cols
        self.target_variable = target_variable
        self.destination_directory = destination_directory

    def load_data(self, data_path: str, date_cols: list, index_col=None):
        """
        Loads the data from a CSV file.

        Args:
            data_path (str): The path to the data file.
            date_cols (list): The list of columns to parse as dates.
            index_col: The column to set as the index (default: None).
        """
        self.data = pd.read_csv(data_path, parse_dates=date_cols, index_col=index_col
                                )
        print(f"Data loaded from {data_path}")

    def save_data(self, destination_directory: str, file_name: str):
        """
        Saves the data to a CSV file.

        Args:
            destination_directory (str): The directory to save the data.
            file_name (str): The name of the file to save the data as.
        """
        os.makedirs(destination_directory, exist_ok=True)
        self.data.to_csv(destination_directory + '/' + file_name)
        print(f"Data saved to {destination_directory}")

    @abstractmethod
    def execute(self):
        """
        Executes the intermediate step.
        """
        pass