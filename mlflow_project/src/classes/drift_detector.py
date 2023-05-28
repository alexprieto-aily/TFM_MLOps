from scipy import stats
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from classes.splitter import Splitter
from classes.classifier_trainer import ClassifierTrainer

class DriftDetector():
    def __init__(
            self
            , name : str
            , random_state : int
    ):
        """Initializes the DriftDetector.

        Args:
            name (str): The name of the drift detector.
            random_state (int): The random state for reproducibility. """
        self.name = name
        self.random_state = random_state

    @staticmethod
    def _run_test_each_column(sample_1 : pd.DataFrame
                              , sample_2 : pd.DataFrame
                              , test_function
                              , alpha =0.05 ):
        """
        Runs a statistical test for each column in two samples.

        Args:
            sample_1 (pd.DataFrame): The first sample.
            sample_2 (pd.DataFrame): The second sample.
            test_function (function): The statistical test function to apply.
            alpha (float): The significance level for the test.

        Returns:
            list: List of columns where the null hypothesis is rejected.
        """
        drifted_columns = []
        for col in sample_1.columns:
            p_value = test_function(sample_1[col], sample_2[col])
            if p_value < alpha:
           
                drifted_columns.append(col)
        return drifted_columns
    
    @staticmethod
    def kolmogorow_smirnov_test(sample_1 : pd.DataFrame
                                , sample_2 : pd.DataFrame) -> float:
        """
        Performs the Kolmogorov-Smirnov test on two samples.

        Args:
            sample_1 (array-like): The first sample.
            sample_2 (array-like): The second sample.

        Returns:
            float: The p-value of the test.
        """
        results = stats.ks_2samp(sample_1, sample_2)
        return results.pvalue
    
    @staticmethod
    def chi_square_test(sample_1 : pd.DataFrame
                        , sample_2 :  pd.DataFrame) -> float:
        """
        Performs the chi-square test on two samples.

        Args:
            sample_1 (array-like): The first sample.
            sample_2 (array-like): The second sample.

        Returns:
            float: The p-value of the test.
        """
        results = stats.chi2_contingency(sample_1, sample_2)
        return results.pvalue

    def univariate_input_drift(self
                               , X_test : pd.DataFrame
                               , X_train: pd.DataFrame) -> list:
        """
        Detects univariate input drift between two datasets.

        Args:
            X_test (pd.DataFrame): The test dataset.
            X_train (pd.DataFrame): The train dataset.

        Returns:
            list: List of columns where drift is detected.
        """
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

    @staticmethod
    def predict_by_period(start_period : int
                        , end_period : int
                        , step : int
                        , model_prod: object
                        , model_challenger: object
                        , objective_metric : str
                        , splitter: Splitter
                        ):
        """
        Makes predictions by period using two models and evaluates their metrics.

        Args:
            start_period (int): The starting period.
            end_period (int): The ending period.
            step (int): The step size.
            model_prod: The production model.
            model_challenger: The challenger model.
            objective_metric (str): The metric to evaluate.
            splitter: The data splitter.

        Returns:
            dict: Dictionary containing metrics for each period.
        """

        metrics = {}
        while start_period < end_period:
            X_next_period, y_next_period = splitter.x_y_filter_by_month(from_month=start_period, to_month=start_period + step)

            if len(X_next_period) == 0:
                print(f"Period {start_period} not found in dataset")
                break
            y_preds_prod = model_prod.predict(X_next_period)
            y_preds_challenger = model_challenger.predict(X_next_period)
            metrics_prod = ClassifierTrainer.get_metrics(y_next_period, y_preds_prod)
            metrics_challenger = ClassifierTrainer.get_metrics(y_next_period, y_preds_challenger)
        
            metrics[start_period] = {'Production model': metrics_prod, 'Challenger model': metrics_challenger}
            print(f"Production model {objective_metric}: {metrics_prod[objective_metric]}")
            print(f"Challenger model {objective_metric}: {metrics_challenger[objective_metric]}\n")
            start_period += step

        return dict(sorted(metrics.items()))
    
    @staticmethod
    def plot_metric(metrics : dict
                    , objective_metric : str
                    ) -> None:
        """
        Plots a metric over time for two models.

        Args:
            metrics (dict): Dictionary containing metrics for each period.
            objective_metric (str): The metric to plot.
        """
        plt.plot(list(metrics.keys()), [x['Production model'][objective_metric] for x in metrics.values()], label='Production model')
        plt.plot(list(metrics.keys()), [x['Challenger model'][objective_metric] for x in metrics.values()], label='Challenger model')
        plt.xlabel('Month')
        plt.ylabel(objective_metric)
        plt.title(f'{objective_metric} by month for production and challenger models')
        plt.legend()
        plt.show()