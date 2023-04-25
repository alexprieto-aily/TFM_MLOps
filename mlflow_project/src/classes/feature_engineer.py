import pandas as pd
import os
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler




from classes.step import Step

class FeatureEngineer(Step):
    def __init__(self
                 , name
                 , data_path
                 , date_cols
                 , true_labels
                 , target_variable
                 , destination_directory
                 ):
        super().__init__(name)
        self.data = None
        self.data_path = data_path
        self.date_cols = date_cols
        self.true_labels = true_labels
        self.target_variable = target_variable
        self.destination_directory = destination_directory
       

    def load_data(self, data_path, date_cols):
            self.data = pd.read_csv(data_path,parse_dates=date_cols)
            print(f"Data loaded from {data_path}")


    def _split_date_columns(self):

        date_cols = self.data.select_dtypes(include='datetime64').columns

        for col in date_cols:
            # self.data[col + '_day'] = self.data[col].dt.day
            self.data[col + '_month'] = self.data[col].dt.month
            self.data[col + '_year'] = self.data[col].dt.year
            print(f"Splitting {col} into {col + '_month'} and {col + '_year'}")
        self.data.drop(columns=date_cols, inplace=True)

    def _fill_missing_values(self, exclude_cols = [None]):
        """
        Function to fill missing values in a dataframe.
        Numeric columns are filled with their median value.
        Categorical columns are filled with the string 'missing'.
        """
        for col in self.data.columns:
            if col not in exclude_cols:
                if self.data[col].dtype == 'object':
                    self.data[col].fillna('Missing', inplace=True)
                else:
                    self.data[col].fillna(self.data[col].median(), inplace=True)
        print(f"Missing values filled in columns {self.data.columns}")

    def _binarize_target(self, true_labels):
        self.data[self.target_variable] = self.data[self.target_variable].isin(true_labels)
        print(f"Target variable {self.target_variable} binarized (1 = {true_labels})")

    def _one_hot_encode(self):

        cat_cols = self.data.select_dtypes(include=['object']).columns

        if len(cat_cols) == 0:
            print("No categorical columns to encode")
            return
        
        if self.target_variable in cat_cols:
            cat_cols.drop(self.target_variable)

        # Convert categorical columns to one-hot encoding
        self.data = pd.get_dummies(self.data, columns=cat_cols, dummy_na=False)
        print(f"Columns encoded: {cat_cols}")

    
    def _standardize_dataframe(self):
        # Find columns with numeric data types
        numeric_cols = self.data.select_dtypes(include=[int, float]).columns.tolist()

        # Standardize numeric columns using Z-score normalization
        scaler = StandardScaler()
        self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])

        # Convert boolean columns to 0s and 1s
        bool_cols = self.data.select_dtypes(include=bool).columns.tolist()
        self.data[bool_cols] = self.data[bool_cols].astype(int)

    def save_data(self, destination_directory):
            os.makedirs(destination_directory, exist_ok=True)
            self.data.to_csv(destination_directory + '/fe_data.csv', index=False)
            print(f"Data saved to {destination_directory}")

    def execute(self):
        self.load_data(self.data_path, self.date_cols)
        self._split_date_columns()
        self._fill_missing_values(exclude_cols=[self.target_variable])
        self._binarize_target(self.true_labels)
        self._one_hot_encode()
        self._standardize_dataframe()
        self.save_data(self.destination_directory)
