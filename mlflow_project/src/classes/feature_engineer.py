import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from classes.intermediate_step import IntermediateStep

class FeatureEngineer(IntermediateStep):
    def __init__(
        self,
        name: str,
        data_path: str,
        date_cols: list,
        true_labels: list,
        target_variable: str,
        destination_directory: str
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
        self.target_variable = target_variable
        self.destination_directory = destination_directory

    def _split_date_columns(self):
        """
        Splits the date columns into month and year columns.
        """
        date_cols = self.data.select_dtypes(include='datetime64').columns

        for col in date_cols:
            self.data[col + '_month'] = self.data[col].dt.month
            self.data[col + '_year'] = self.data[col].dt.year
            print(f"Splitting {col} into {col + '_month'} and {col + '_year'}")
        self.data.drop(columns=date_cols, inplace=True)

    def _fill_missing_values(self, exclude_cols: list = [None]):
        """
        Fills missing values in the dataframe.

        Numeric columns are filled with their median value.
        Categorical columns are filled with the string 'Missing'.

        Args:
            exclude_cols (list): List of columns to exclude from filling missing values.
        """
        for col in self.data.columns:
            if col not in exclude_cols:
                if self.data[col].dtype == 'object':
                    self.data[col].fillna('Missing', inplace=True)
                else:
                    self.data[col].fillna(self.data[col].median(), inplace=True)
        print(f"Missing values filled in columns {self.data.columns}")

    def _binarize_target(self, true_labels: list):
        """
        Binarizes the target variable based on true labels.

        Args:
            true_labels (list): List of true labels for the target variable.
        """
        self.data[self.target_variable] = self.data[self.target_variable].isin(
            true_labels)
        print(
            f"Target variable {self.target_variable} binarized (1 = {true_labels})")

    def _one_hot_encode(self):
        """
        Performs one-hot encoding on categorical columns.
        """
        cat_cols = self.data.select_dtypes(include=['object']).columns

        if len(cat_cols) == 0:
            print("No categorical columns to encode")
            return

        if self.target_variable in cat_cols:
            cat_cols.drop(self.target_variable)

        self.data = pd.get_dummies(self.data, columns=cat_cols, dummy_na=False)
        print(f"Columns encoded: {cat_cols}")

    def _standardize_dataframe(self):
        """
        Standardizes numeric columns using Z-score normalization.
        Converts boolean columns to 0s and 1s.
        """
        numeric_cols = self.data.select_dtypes(
            include=[int, float]).columns.tolist()

        scaler = StandardScaler()
        self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])

        bool_cols = self.data.select_dtypes(include=bool).columns.tolist()
        self.data[bool_cols] = self.data[bool_cols].astype(int)

    def execute(self):
        """
        Executes the feature engineering process.
        """
        print(f"Executing {self.name} step")
        self.load_data(self.data_path, self.date_cols, index_col=0)
        self._split_date_columns()
        self._fill_missing_values(exclude_cols=[self.target_variable])
        self._binarize_target(self.true_labels)
        self._one_hot_encode()
        self._standardize_dataframe()
        self.save_data(self.destination_directory, 'fe_data.csv')
        print(f"Finished executing {self.name} step")
