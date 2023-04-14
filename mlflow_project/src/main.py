import utils
from importer import Importer

utils.set_parent_directory_as_working_directory()

# TODO: Move this to a config file
DATA_FOLDER = "./data/"
KAGGLE_DATASET_NAME = "wordsforthewise/lending-club"
RAW_DATA_FOLDER_NAME = "raw"

# Import the data
importer_name = "importer"
importer =Importer(
    name=importer_name
    , data_folder=DATA_FOLDER
    , kaggle_dataset_name=KAGGLE_DATASET_NAME
    , destination_directory_name=RAW_DATA_FOLDER_NAME)

# TODO: remove waw as argument
importer.execute('raw')