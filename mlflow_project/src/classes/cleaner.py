from typing import Union
import pandas as pd
import os
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns


from classes.intermediate_step import IntermediateStep


class Cleaner(IntermediateStep):
    def __init__(
            self
            , name
            , data_path
            , date_cols
            , columns_to_keep
            , destination_directory
            , null_threshold
            , target_variable
            , max_corr
            ):
        super().__init__(name
                         , data_path
                         , date_cols
                         , target_variable
                         , destination_directory)
        self.columns_to_keep = columns_to_keep
        self.destination_directory = destination_directory
        self.null_threshold = null_threshold
        self.max_corr = max_corr

   
    
    def _keep_columns(self, columns_to_keep):
        self.data = self.data[columns_to_keep]
        print(f"Columns kept {columns_to_keep}")

    def check_data(self):
        """
        Check the data types and percentage of null values in a Pandas DataFrame.

        Args:
        ----
            df : pd.DataFrame
                Input DataFrame to check

        Returns:
        -------
            pd.DataFrame
                DataFrame containing data types and null percentages for each column.
        """
        dtypes = self.data.dtypes
        null_percentages = self.data.isnull().mean() * 100
        check_df = pd.DataFrame(
            {'Data Types': dtypes, 'Null Percentages': null_percentages})
        return check_df .join(self.data.head(3).T)

    
    def _drop_columns_nulls(self, null_threshold: float):
        """
        Drops any column in a Pandas DataFrame with a percentage of nulls higher than a given threshold.

        Args:
        ----
            df : pd.DataFrame
                Input DataFrame to check
            threshold : float
                Threshold percentage of null values to drop columns. Must be a float between 0 and 100.

        Returns:
        -------
            pd.DataFrame
                DataFrame with dropped columns.
        """
        if not 0 <=  null_threshold <= 100:
            raise ValueError("Threshold must be a float between 0 and 100.")

        null_percentages = self.data.isnull().mean() * 100
        columns_to_keep = null_percentages[null_percentages <=  null_threshold].index.tolist(
        )
        self.data = self.data[columns_to_keep]
        print(f"Columns with null percentages higher than {null_threshold} dropped.")
    
    def _drop_target_variable_nulls(self, target_variable):
        self.data = self.data.dropna(subset=[target_variable])
        print(f"Rows with null values in {target_variable} dropped.")

    def _drop_high_correlation_vars(self, max_corr):
        corr_matrix = self.data.corr().abs()
        upper = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(
            upper[column] > max_corr)]
        self.data = self.data.drop(to_drop, axis=1)
        print(f"Columns with correlation higher than {max_corr} dropped.")

    def corr_heatmap(self, title=None, figsize=(10, 10), annot=True, cmap='coolwarm', linewidth=.5, fontsize=7):
        plt.figure(figsize=figsize)
        df_corr = self.data.corr()
        df_corr = df_corr.round(2)
        mask = np.triu(np.ones_like(df_corr))
        sns.heatmap(df_corr, annot=annot, cmap=cmap, linewidth=linewidth,
                    mask=mask, annot_kws={"fontsize": fontsize})
        plt.title(title)
        plt.show()

   

    def _create_dates_data(self):
        dates_data = self.data[['term', 'issue_d','loan_status','last_pymnt_d']]
        dates_data = dates_data[~dates_data['issue_d'].isnull()]
        dates_data['term']=dates_data['term'].str.extract('(\d+)')
        dates_data['term']=dates_data['term'].astype(int)
        dates_data['finished_d'] = dates_data.apply(lambda x: x['issue_d'] + pd.DateOffset(months=x['term']), axis=1)
        dates_data.to_csv(self.destination_directory + '/dates_data.csv')
        print(f"Dates data saved to {self.destination_directory}")
            
    def execute(self):
        print(f"Executing {self.name} step")
        self.load_data(self.data_path, self.date_cols, None)
        self._create_dates_data()
        self._keep_columns(self.columns_to_keep)
        self._drop_columns_nulls(self.null_threshold)
        self._drop_target_variable_nulls(self.target_variable)
        self._drop_high_correlation_vars(self.max_corr)
        self.save_data(self.destination_directory, 'clean_data.csv')
        print(f"Finished executing {self.name} step")