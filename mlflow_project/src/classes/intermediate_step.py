import pandas as pd
import os
from classes.step import Step
from abc import abstractmethod

class IntermediateStep(Step):
    def __init__(self
                 , name
                 , data_path
                 , date_cols
                 , target_variable
                 , destination_directory
                 ):
        super().__init__(name)
        self.data = None
        self.data_path = data_path
        self.date_cols = date_cols
        self.target_variable = target_variable
        self.destination_directory = destination_directory


    def load_data(self, data_path, date_cols, index_col=None):
        self.data = pd.read_csv(data_path, parse_dates=date_cols, index_col=index_col
                                )
        print(f"Data loaded from {data_path}")
    
    def save_data(self, destination_directory, file_name):
        os.makedirs(destination_directory, exist_ok=True)
        self.data.to_csv(destination_directory + '/' + file_name)
        print(f"Data saved to {destination_directory}")

    @abstractmethod
    def execute(self):
        pass
