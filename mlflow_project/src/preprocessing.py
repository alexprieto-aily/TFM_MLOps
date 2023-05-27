# Importer
import classes.utils as utils
from classes.importer import Importer
from classes.cleaner import Cleaner
from classes.feature_engineer import FeatureEngineer

# Cleaner
import pandas as pd
import os

# Set the working directory to the parent directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))
utils.set_parent_directory_as_working_directory()

# TODO: Move this to a config file
# Importing
DATA_FOLDER = "./data"
KAGGLE_DATASET_NAME = "wordsforthewise/lending-club" 
RAW_DATA_FOLDER = DATA_FOLDER + "/raw"

# Cleaning
RAW_DATA_PATH  = RAW_DATA_FOLDER + "/accepted_2007_to_2018Q4.csv"

# Feature Engineering
CLEAN_DATA_PATH = DATA_FOLDER +'/clean_data.csv'

# Variables
importer_name = "importer"
cleaner_name = "cleaner"
feature_engineer_name = "feature_engineer"

columns_to_keep =  pd.read_csv('src/available_features.csv')
columns_to_keep = columns_to_keep['features'].tolist()
date_cols=['issue_d']

target_variable='loan_status'
true_labels = ['Fully Paid', 'Current', 'Does not meet the credit policy. Status:Fully Paid', 'In Grace Period']




# Import the data

importer = Importer(
    name=importer_name
    , data_folder=DATA_FOLDER
    , kaggle_dataset_name=KAGGLE_DATASET_NAME
    , destination_directory=RAW_DATA_FOLDER
    )


# Clean the data
cleaner = Cleaner(
    name=cleaner_name
    , data_path = RAW_DATA_PATH 
    , date_cols = date_cols
    , columns_to_keep = columns_to_keep
    , destination_directory = DATA_FOLDER
    , null_threshold=30
    , target_variable=target_variable
    , max_corr=0.75)

# Feature engineer the data
feature_engineer = FeatureEngineer(
    name = feature_engineer_name
    , data_path = CLEAN_DATA_PATH
    , date_cols = date_cols
    , true_labels = true_labels
    , target_variable = target_variable
    , destination_directory = DATA_FOLDER
)

if __name__ == "__main__":

    importer.execute()
    cleaner.execute()
    feature_engineer.execute()