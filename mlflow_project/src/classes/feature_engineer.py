import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA



from classes.step import Step

class Feature_engineer(Step):
    def __init__(self, name):
        super().__init__(name)
        self.data = None
        self.target_variable = None



    def _split_date_columns(self):

        date_cols = self.data.select_dtypes(include='datetime64').columns

        for col in date_cols:
            # self.data[col + '_day'] = self.data[col].dt.day
            self.data[col + '_month'] = self.data[col].dt.month
            self.data[col + '_year'] = self.data[col].dt.year
        self.data  = self.data.drop(columns=date_cols, inplace=True)





    def _impute_missing_knn(self, target_variable, k=5):
        """
        Imputes missing values in a dataset using the KNN algorithm.

        Parameters:
        df (pandas.DataFrame): The input dataset with missing values.
        k (int): The number of nearest neighbors to use for imputation (default=5).

        Returns:
        pandas.DataFrame: The imputed dataset with no missing values.
        """
        y = self.data[target_variable]
        self.data.drop(columns=target_variable, inplace=True)

        cont_vars = self.data.select_dtypes(include=['number']).columns.tolist()
        cat_vars = self.data.select_dtypes(include='object').columns.tolist()

        imputer = KNNImputer(n_neighbors=k)
        self.data.loc[:, cont_vars] = imputer.fit_transform(self.data[cont_vars])

        # Impute missing values in categorical variables using 'Missing' word
        self.data.loc[:,  cat_vars] = self.data.loc[:,  cat_vars].fillna('Missing')

        return pd.concat([df, y], axis=1)


    def _one_hot_encode(self):

        cat_cols = self.data.select_dtypes(include=['object']).columns

        # Convert categorical columns to one-hot encoding
        self.data = pd.get_dummies(self.data, columns=cat_cols, dummy_na=False)

     


    def standardize_dataframe(self):
        # Find columns with numeric data types
        numeric_cols = self.data.select_dtypes(include=[int, float]).columns.tolist()

        # Standardize numeric columns using Z-score normalization
        scaler = StandardScaler()
        self.data[numeric_cols] = scaler.fit_transform(self.data[numeric_cols])

        # Convert boolean columns to 0s and 1s
        bool_cols = self.data.select_dtypes(include=bool).columns.tolist()
        self.data = self.data[bool_cols] = self.data[bool_cols].astype(int)
