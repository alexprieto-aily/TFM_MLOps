from classes.intermediate_step import IntermediateStep
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import pandas as pd
import os


class Splitter(IntermediateStep):
    def __init__(
            self,
            name: str,
            data_path: str,
            date_cols: list,
            target_variable: str,
            destination_directory: str,
            dates_data_path: str,
            column_to_split_by: str,
            test_size: float,
            random_state: int
    ):
        """
        Initializes the Splitter object.

        Args:
            name (str): The name of the splitter.
            data_path (str): The path to the data file.
            date_cols (list): The list of date columns in the data.
            target_variable (str): The name of the target variable.
            destination_directory (str): The directory to save the train and test data.
            dates_data_path (str): The path to the dates data file.
            column_to_split_by (str): The column to split the data by.
            test_size (float): The proportion of the data to include in the test set.
            random_state (int): The random state for reproducibility.
        """
        super().__init__(name, data_path, date_cols, target_variable, destination_directory)
        self.dates_data_path = dates_data_path
        self.column_to_split_by = column_to_split_by
        self.dates_data = None
        self.test_size = test_size
        self.random_state = random_state
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None

    @staticmethod
    def resample_data(X: pd.DataFrame, y: pd.Series, random_state: int, over_sampling_strategy: float = 0.5,
                      under_sampling_strategy: float = 0.8) -> tuple:
        """
        Resamples the data using SMOTE for oversampling and RandomUnderSampler for undersampling.

        Args:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The target variable.
            random_state (int): The random state for reproducibility.
            over_sampling_strategy (float): The sampling strategy for oversampling.
            under_sampling_strategy (float): The sampling strategy for undersampling.

        Returns:
            tuple: The resampled feature matrix and target variable.
        """
        smt = SMOTE(sampling_strategy=over_sampling_strategy, random_state=random_state)
        X, y = smt.fit_resample(X, y)
        under = RandomUnderSampler(sampling_strategy=under_sampling_strategy)
        X, y = under.fit_resample(X, y)
        return X, y

    def set_train_test(self, X: pd.DataFrame, y: pd.Series, test_size: float, resample: bool = True):
        """
        Splits the data into train and test sets.

        Args:
            X (pd.DataFrame): The feature matrix.
            y (pd.Series): The target variable.
            test_size (float): The proportion of the data to include in the test set.
            resample (bool): Whether to resample the data or not.
        """
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X
                                                                                , y
                                                                                , test_size=test_size
                                                                                , random_state=self.random_state)
        
        if resample:
            self.X_train, self.y_train = self.resample_data(self.X_train
                                                            , self.y_train
                                                            , random_state=self.random_state)
            
            print(f"Resampled data. New train size: {len(self.X_train)}")

        print(f"""Test and train attributes defined {test_size}.
        Test size: {len(self.X_test)}
        Train size: {len(self.X_train)}""")

    def save_data(self, destination_directory: str):
        """
        Saves the train and test data to the destination directory.

        Args:
            destination_directory (str): The directory to save the train and test data.
        """
        os.makedirs(destination_directory, exist_ok=True)
        self.X_train.to_csv(destination_directory + '/X_train.csv')
        self.X_test.to_csv(destination_directory + '/X_test.csv')
        self.y_train.to_csv(destination_directory + '/y_train.csv')
        self.y_test.to_csv(destination_directory + '/y_test.csv')
        print(f"Data saved to {destination_directory}")

    def execute(self):
        """
        Executes the splitter.
        """
        print(f"-------------- Executing {self.name} --------------")
        self.load_data(self.data_path, self.date_cols, index_col=0)
        self.data.columns= self.data.columns.str.replace(' ', '_')
        self.load_dates_data(self.dates_data_path, self.date_cols, index_col=0)  
        self.set_train_test(self.data.loc[:, self.data.columns != self.target_variable]
                            , self.data[self.target_variable]
                            , self.test_size
                            , resample=False
                     )
        #self.save_data(self.destination_directory)
        print(f"--------------- {self.name} finished ---------------")

    # These methods below are specific for this use case
    def load_dates_data(self, data_path: str, date_cols: list, index_col: str = None):
        """
        Loads the dates data from the given path.

        Args:
            data_path (str): The path to the dates data file.
            date_cols (list): The list of date columns in the data.
            index_col (str): The column to be used as the index.

        Returns:
            pd.DataFrame: The dates data.
        """
        self.dates_data = pd.read_csv(data_path, parse_dates=date_cols, index_col=index_col
                                      )
        print(f"Dates data loaded from {data_path}")

    def _add_date_column_to_data(self, data: pd.DataFrame, dates_data_path: str, column_to_split_by: str) -> pd.DataFrame:
        """
        Adds a column with the date to the data.

        Args:
            data (pd.DataFrame): The data to add the date column to.
            dates_data_path (str): The path to the dates data file.
            column_to_split_by (str): The column to split the data by.

        Returns:
            pd.DataFrame: The data with the added date column.
        """
        # Load the dates data
        dates_data = pd.read_csv(dates_data_path, index_col=0)
        dates_data = dates_data[[column_to_split_by]]
        data = data.merge(
            dates_data, how='left', left_index=True, right_index=True)
        data[self.column_to_split_by] = data[self.column_to_split_by].astype(
            'datetime64[ns]')
        return data

    def _filter_by_month(self, data: pd.DataFrame, from_month: int, to_month: int,
                         column_to_split_by: str) -> pd.DataFrame:
        """
        Filters the data from the specified months.

        Args:
            data (pd.DataFrame): The data to be filtered.
            from_month (int): The starting month.
            to_month (int): The ending month.
            column_to_split_by (str): The column to split the data by.

        Returns:
            pd.DataFrame: The filtered data.
        """
        cutoff_lower = data[column_to_split_by].min(
        ) + pd.DateOffset(months=from_month)
        cutoff_upper = data[column_to_split_by].min(
        ) + pd.DateOffset(months=to_month)
        return data[(data[column_to_split_by] >= cutoff_lower) & (data[column_to_split_by] < cutoff_upper)]

    def x_y_filter_by_month(self, from_month: int, to_month: int) -> tuple:
        """
        Returns the X and y data for the specified month range.

        Args:
            from_month (int): The starting month.
            to_month (int): The ending month.

        Returns:
            tuple: The filtered feature matrix and target variable.
        """
        filtered_data = self._add_date_column_to_data(self.data
                                      , self.dates_data_path
                                      , self.column_to_split_by)
        
        filtered_data = self._filter_by_month(filtered_data
                                              , from_month
                                              , to_month
                                              , self.column_to_split_by)
        filtered_data = filtered_data.drop(self.column_to_split_by, axis=1)
        return filtered_data.loc[:, filtered_data.columns != self.target_variable], filtered_data[self.target_variable]

    def set_train_test_filtered(self, number_of_months: int):
        """
        Sets the train and test data for the specified number of months.

        Args:
            number_of_months (int): The number of months to include in the train and test data.
        """
        X, y=  self.x_y_filter_by_month(0, number_of_months)

        self.set_train_test(X
                            , y
                            , self.test_size
                            )
