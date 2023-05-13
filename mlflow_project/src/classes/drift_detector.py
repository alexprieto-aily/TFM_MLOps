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

    @staticmethod
    def _run_test_each_column( sample_1, sample_2, test_function, alpha=0.05):
        drifted_columns = []
        print(f"{test_function.__name__} test")
        for col in sample_1.columns:
            p_value = test_function(sample_1[col], sample_2[col])
            if p_value < alpha:
                print(f"Warning: {col} has drifted")
                drifted_columns.append(col)
        return drifted_columns
    
    @staticmethod
    def kolmogorow_smirnov_test(sample_1, sample_2):
        results = stats.ks_2samp(sample_1, sample_2)
        return results.pvalue
    
    @staticmethod
    def chi_square_test(sample_1, sample_2):
        results = stats.chi2_contingency(sample_1, sample_2)
        return results.pvalue
    

    def univariate_input_drift(self, X_test, X_train):
        numerical_columns = X_test.select_dtypes(include=np.number).columns
        categorical_columns = X_test.select_dtypes(include='object').columns

        drifted_columns = []

        if len(numerical_columns)>0:
            numerical_drifted_columns = self._run_test_each_column(X_test[numerical_columns]
                                                               , X_train[numerical_columns]
                                                               , self.kolmogorow_smirnov_test)
            drifted_columns.append(numerical_drifted_columns)
        
        if len(categorical_columns)>0:
            categorical_drifted_columns = self._run_test_each_column(X_test[categorical_columns]
                                                                , X_train[categorical_columns]
                                                                , self.chi_square_test)
            drifted_columns.append(categorical_drifted_columns)

        return drifted_columns

