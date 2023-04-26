from classes.intermediate_step import IntermediateStep
from sklearn.model_selection import train_test_split

import pandas as pd

class Splitter(IntermediateStep):
    def __init__(
            self
            , name
            , data_path
            , date_cols
            , target_variable
            , destination_directory
            , dates_data_path
            , column_to_split_by
            , number_of_months
            , test_size
            , random_state      
            ):

        super().__init__(name
                    , data_path
                    , date_cols
                    , target_variable
                    , destination_directory)
        self.dates_data_path = dates_data_path
        self.column_to_split_by = column_to_split_by
        self.dates_data = None
        self.test_size = test_size
        self.random_state = random_state
        self.train = None
        self.test = None

    def load_dates_data(self, data_path, date_cols, index_col=None):
        self.dates_data = pd.read_csv(data_path, parse_dates=date_cols, index_col=index_col
                                )
        print(f"Dates data loaded from {data_path}")
    
    def _add_date_column_to_data(self, dates_data_path, column_to_split_by):
        """This function adds a column with the date to the data
        """
        # Load the dates data
        dates_data = pd.read_csv(dates_data_path, index_col=0)
        dates_data =  dates_data[[column_to_split_by]]
        self.data = self.data.merge(dates_data, how='left', left_index=True, right_index=True)
        self.data[self.column_to_split_by] = self.data[self.column_to_split_by].astype('datetime64[ns]')
        print(f"Date column {column_to_split_by} added to the data")

   
    def _filter_by_month(self, number_of_months, column_to_split_by):
        """This function filters the data by the number of months
        """
        cutoff = self.data[column_to_split_by].min() + pd.DateOffset(months=number_of_months)
        return self.data[self.data[column_to_split_by] < cutoff]
      

    def define_test_train(self, data,  test_size, random_state):
        self.train, self.test = train_test_split(data
                                                 , test_size=test_size
                                                 , random_state=random_state
                                                 )
        print(f"""Test and train attributes defined {test_size}.
        Test size: {len(self.test)}
        Train size: {len(self.train)}""")

    def split_data_filtered(self, number_of_months):
        filtered_data = self._filter_by_month(number_of_months, self.column_to_split_by)
        filtered_data.drop(columns=self.column_to_split_by, inplace=True)
        self.define_test_train(filtered_data, self.test_size, self.random_state)

    def execute(self):
        self.load_data(self.data_path, self.date_cols, index_col=0)
        self.load_dates_data(self.dates_data_path, ['issue_d','last_pymnt_d' ,'finished_d'], index_col=0)
        self._add_date_column_to_data(self.dates_data_path, self.column_to_split_by)


    