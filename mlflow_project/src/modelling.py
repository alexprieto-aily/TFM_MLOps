# Importer
import classes.utils as utils
from classes.splitter import Splitter
from classes.classifier_trainer import ClassifierTrainer
from classes.drift_detector import DriftDetector
from classes.mlflow_manager import MLflowManager

from sklearn.tree import DecisionTreeClassifier


import numpy as np


utils.set_parent_directory_as_working_directory()

# TODO: Move this to a config file
# Importing
DATA_FOLDER = "./data"


FE_DATA_PATH = DATA_FOLDER +'/fe_data.csv'
DATES_DATA_PATH = DATA_FOLDER +'/dates_data.csv'

SEED = 47

import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier

experiment_name = 'TEST_decision_tree_whole_timeseries'

splitter = Splitter(
    name = "splitter"
    , data_path = FE_DATA_PATH
    , date_cols = []
    , target_variable = 'loan_status'
    , destination_directory = DATA_FOLDER
    , dates_data_path = DATES_DATA_PATH
    , column_to_split_by = 'finished_d'
    , test_size = 0.3
    , random_state = SEED
)
splitter.execute()

mlflow_manager =  MLflowManager(
    name = 'mlflow_manager' 
)

def choose_prod_challenger_model(prod, challenger, X, y,  objective_metric):

    prod_model_preds = prod.predict(X)
    challenger_model_preds = challenger.predict(X)

    prod_model_metrics = ClassifierTrainer.get_metrics(y, prod_model_preds)
    challenger_model_metrics = ClassifierTrainer.get_metrics(y, challenger_model_preds)

    is_challenger =  prod_model_metrics[objective_metric] < challenger_model_metrics[objective_metric]
    if is_challenger:
        print(f'- Challenger {objective_metric}: {challenger_model_metrics[objective_metric]}')
        print(f'- Prod {objective_metric}: {prod_model_metrics[objective_metric]}')
        print('Challenger model is better!')
    else:
        print(f'- Challenger {objective_metric}: {challenger_model_metrics[objective_metric]}')
        print(f'- Prod {objective_metric}: {prod_model_metrics[objective_metric]}')
        print('Prod model is better!')
    
    return  prod_model_metrics, challenger_model_metrics, is_challenger

def input_target_drift(splitter, step, months):

    drift_detector = DriftDetector(
    name = 'drift_detector'
    , random_state=SEED
    )

    X_first_period,  y_first_period = splitter.x_y_filter_by_month(from_month=months-step*2, to_month=months-step)
    X_second_period,  y_second_period = splitter.x_y_filter_by_month(from_month=months-step, to_month=months)

    drifted_columns = drift_detector.univariate_input_drift(X_first_period, X_second_period)
    print(f'Drifted variables: {drifted_columns}')

    p_value = drift_detector.kolmogorow_smirnov_test(y_first_period, y_second_period)

    if p_value < 0.05:
        print("Warning: target variable has drifted")
    



def run_whole_timeseries(params
                         , splitter
                         , mlflow_manager
                         , step
                         , random_state
                         , model_class
                         , objective_metric= 'roc_auc'):

    start_date = splitter.dates_data['finished_d'].min()
    start_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date = splitter.dates_data['finished_d'].max()
    end_date = datetime.strptime(end_date, '%Y-%m-%d')
    prod_model_names = []
    months = step
    current_iteration = 1

    while start_date < end_date:

        print(f"""------------------ Iteration {current_iteration} started ------------------""")    

        print(f"""Training model for {start_date} to {start_date + pd.DateOffset(months=step)}""")
        splitter.set_train_test_filtered(number_of_months=months)

        # Train new model
        model_name = f"model_{current_iteration}_year"
        trainer =  ClassifierTrainer(
            name = model_name
            , model_class = model_class
            , random_state=random_state
            , splitter = splitter
            , objective_metric = objective_metric
        )

        trainer.set_model_params(params)
        trainer.train()
        trainer.predict(splitter.X_test)
        trainer.evaluate(splitter.y_test)

        

        # After first year
        if current_iteration>=2:

            # Input and target drift
            input_target_drift(splitter, step, months)

            # Concept drift
            X_next_year, y_next_year = splitter.x_y_filter_by_month(from_month=months, to_month=months+step)

            assert len(X_next_year) > 0 , 'No more data for next year'

            prod = MLflowManager.load_model(prod_model_names[-1], '1', 'http://localhost:5000')
            challenger = trainer.model_class

            prod_model_metrics, challenger_model_metrics, is_challenger = choose_prod_challenger_model(prod
                                                                             , challenger
                                                                             , X_next_year
                                                                             , y_next_year
                                                                             , objective_metric
                                                                             )

            # Setting production model
            if is_challenger:
                log_model = True       
                prod_model_names.append(trainer.name)
            else:
                log_model = False

            # Log metrics
            run_name = f"run_{current_iteration}_year"
            mlflow_manager.make_run(run_name=run_name
                        , experiment_name=experiment_name
                        , trainer=trainer
                        , log_model=log_model
                        )

            mlflow_manager.log_new_metrics(run_name= run_name
                            , experiment_name=experiment_name
                            , metrics= prod_model_metrics
                            , prefix='Production_next_year_'
                            )
            
            mlflow_manager.log_new_metrics(run_name= run_name
                                        , experiment_name=experiment_name
                                        , metrics= challenger_model_metrics
                                        , prefix='Challenger_next_year_'
                                        )

            if is_challenger:
                MLflowManager.set_model_stage(prod_model_names[-1], model_version=1, stage='production')
                MLflowManager.set_model_stage(prod_model_names[-2], model_version=1, stage='archived')


        # First year
        else:
            
            run_name = f"run_{current_iteration}_year"

            mlflow_manager.make_run(run_name = run_name
                          , experiment_name=experiment_name
                          , trainer=trainer
                          , log_model=True
                           )
                           
            prod_model_names.append(trainer.name)
            MLflowManager.set_model_stage(trainer.name, model_version=1, stage='production')

        months += step  
        start_date = start_date + pd.DateOffset(months=step)
        print(f"""------------------ Iteration {current_iteration} finished ------------------\n""")
        current_iteration += 1 

    return prod_model_metrics, prod_model_names

if __name__ == "__main__":
    prod_model_metrics, prod_model_names = run_whole_timeseries(
                        params={'max_depth': 7}
                        , splitter=splitter
                        , mlflow_manager=mlflow_manager
                        , step=12
                        , random_state=SEED
                        , model_class=DecisionTreeClassifier()
                        )
