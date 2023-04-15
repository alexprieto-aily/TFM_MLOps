import os
import zipfile
import shutil
import kaggle


from classes.step import Step


class Importer(Step):
    def __init__(self, name, data_folder, kaggle_dataset_name, destination_directory):
        super().__init__(name)
        self.data_folder = data_folder
        self.kaggle_dataset_name = kaggle_dataset_name
        self.destination_directory = destination_directory

    def _download_data_kaggle(self):
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
            self.kaggle_dataset_name,
            path=self.data_folder,
        )

        zip_name = self.kaggle_dataset_name.split("/")[1] + ".zip"
        zip_name = os.path.join(
            self.data_folder, zip_name
        )

        # Extract all the files to the destination folder
        with zipfile.ZipFile(zip_name, "r") as zip_ref:
            zip_ref.extractall(self.data_folder)

        os.remove(zip_name)
        print(f"Data downloaded to {self.data_folder}")

    def move_csv_files_to_directory(self, destination_directory):
        
        os.makedirs(destination_directory, exist_ok=True)

        # Walk through the directory tree and move all CSV files to the 'raw' directory
        for root, dirs, files in os.walk(self.data_folder):
            for filename in files:
                if filename.endswith('.csv'):
                    filepath = os.path.join(root, filename)
                    new_filepath = os.path.join(
                        destination_directory, filename)
                    shutil.move(filepath, new_filepath)

        print(
            f"All CSV files in {self.data_folder} have been moved to {destination_directory}")

    def delete_except_directories(self, directories_to_keep):
        # Convert directories_to_keep to a set for faster lookup
        directories_to_keep = [os.path.join(
            self.data_folder, directory_to_keep) for directory_to_keep in directories_to_keep]

        # Walk through the directory tree and delete all files and directories except for directories_to_keep
        for root, dirs, files in os.walk(self.data_folder, topdown=False):
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

        print(
            f"All files and directories in {self.data_folder} except for {directories_to_keep} have been deleted")

    def get_files_created(self):
        return [os.path.join(self.data_folder, "raw")]

    def execute(self):

        # Download the data from Kaggle
        self._download_data_kaggle()

        # Move all CSV files to the 'raw' directory
        self.move_csv_files_to_directory(self.destination_directory)

        # Delete all files and directories except for the 'raw' directory
        self.delete_except_directories([self.destination_directory])
