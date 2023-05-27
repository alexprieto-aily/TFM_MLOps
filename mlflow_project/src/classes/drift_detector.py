from scipy import stats
import numpy as np
import matplotlib.pyplot as plt

from classes.splitter import Splitter
from classes.classifier_trainer import ClassifierTrainer

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
        for col in sample_1.columns:
            p_value = test_function(sample_1[col], sample_2[col])
            if p_value < alpha:
           
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

    @staticmethod
    def predict_by_period(start_period
                        , end_period
                        , step
                        , model_prod
                        , model_challenger
                        , objective_metric
                        , splitter):

        metrics = {}
        while start_period < end_period:
            X_next_period, y_next_period = splitter.x_y_filter_by_month(from_month=start_period, to_month=start_period + step)
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
    def plot_metric(metrics, objective_metric):
        plt.plot(list(metrics.keys()), [x['Production model'][objective_metric] for x in metrics.values()], label='Production model')
        plt.plot(list(metrics.keys()), [x['Challenger model'][objective_metric] for x in metrics.values()], label='Challenger model')
        plt.xlabel('Month')
        plt.ylabel(objective_metric)
        plt.title(f'{objective_metric} by month for production and challenger models')
        plt.legend()
        plt.show()