from scipy import stats
import numpy as np
from classes.splitter import Splitter

class DriftDetector():
    def __init__(
            self
            , name
            , random_state
    ):
        self.name = name
        self.random_state = random_state


    def kolmogorow_smirnov_test(self, X_test, X_train):

        drifted_columns = []
        print("Kolmogorov-Smirnov test")
        for col in X_test.columns:
            results = stats.ks_2samp(X_test[col], X_train[col])
            if results.pvalue < 0.05:
                print(f"Warning: {col} has drifted")
                drifted_columns.append(col)
        return drifted_columns
    
    
    def chi_square_test(self, X_test, X_train):

        drifted_columns = []
        print("Chi square test")
        for col in X_test.columns:
            p_value = stats.chi2_contingency(X_test[col], X_train[col])
            if p_value < 0.05:
                print(f"Warning: {col} has drifted")
                drifted_columns.append(col)
        return drifted_columns


    def univariate_input_drift(self, X_test, X_train):
        numerical_columns = X_test.select_dtypes(include=np.number).columns
        categorical_columns = X_test.select_dtypes(include='object').columns

        numerical_drifted_columns = self.kolmogorow_smirnov_test(X_test[numerical_columns], X_train[numerical_columns])
        categorical_drifted_columns = self.chi_square_test(X_test[categorical_columns], X_train[categorical_columns])

        return numerical_drifted_columns + categorical_drifted_columns

