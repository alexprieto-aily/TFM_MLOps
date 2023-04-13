import os
import zipfile
import shutil
import kaggle


def set_parent_directory_as_working_directory():
    """This function sets the parent directory as the working directory."""

    current_dir = os.path.abspath('.')
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)


def download_data_kaggle(
        destination_directory,
        kaggle_dataset_name
        ):

    # Check if the Kaggle API key was created
    if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
        raise Exception(
            "Kaggle API key not found. \
            Make sure to follow the instructions to set up your\
            Kaggle API key on \
            https://www.kaggle.com/docs/api#getting-started-installation-&-authentication"
        )

    # Download the dataset into a current folder
    kaggle.api.dataset_download_files(
        kaggle_dataset_name,
        path=destination_directory,
    )

    zip_name = kaggle_dataset_name.split("/")[1] + ".zip"
    zip_name = os.path.join(
        destination_directory, zip_name
    )

    # Extract all the files to the destination folder
    with zipfile.ZipFile(zip_name, "r") as zip_ref:
        zip_ref.extractall(destination_directory)

    os.remove(zip_name)
    print(f"Data downloaded to {destination_directory}")



def move_csv_files_to_directory(
        directory,
        destination_directory_name):
    # Create a new directory called 'raw' inside the given directory
    destination_directory = os.path.join(directory, destination_directory_name)
    os.makedirs(destination_directory, exist_ok=True)
    
    # Walk through the directory tree and move all CSV files to the 'raw' directory
    for root, dirs, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.csv'):
                filepath = os.path.join(root, filename)
                new_filepath = os.path.join(destination_directory, filename)
                shutil.move(filepath, new_filepath)
    
    print(f"All CSV files in {directory} have been moved to {destination_directory}")


def delete_except_directories(
        directory,
        directories_to_keep):
    # Convert directories_to_keep to a set for faster lookup

    directories_to_keep = [os.path.join(directory, directory_to_keep) for directory_to_keep in directories_to_keep]

    
    # Walk through the directory tree and delete all files and directories except for directories_to_keep
    for root, dirs, files in os.walk(directory, topdown=False):
        # Delete all files that are not in directories_to_keep
        for filename in files:
            filepath = os.path.join(root, filename)
            if root not in directories_to_keep:
                os.remove(filepath)

        # Delete all directories except for directories_to_keep
        for dirname in dirs:
            dirpath = os.path.join(root, dirname)
            if dirpath not in directories_to_keep:
                shutil.rmtree(dirpath)
        
    print(f"All files and directories in {directory} except for {directories_to_keep} have been deleted")

def load_raw_data(directory, kaggle_dataset_name, raw_data_dirname):
    # Set the parent directory as the working directory
    set_parent_directory_as_working_directory()
    
    # Download the data from Kaggle
    download_data_kaggle(directory, kaggle_dataset_name)
    
    # Move all CSV files to the 'raw' directory
    move_csv_files_to_directory(directory, raw_data_dirname)
    
    # Delete all files and directories except for the 'raw' directory
    delete_except_directories(directory, [raw_data_dirname])

load_raw_data(directory=DATA_FOLDER, kaggle_dataset_name=KAGGLE_DATASET_NAME, raw_data_dirname="raw")
