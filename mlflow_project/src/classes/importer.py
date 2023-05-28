import os
import zipfile
import shutil
import kaggle


from classes.step import Step


class Importer(Step):
    def __init__(
        self,
        name: str,
        data_folder: str,
        kaggle_dataset_name: str,
        destination_directory: str
    ):
        """
        Initializes the Importer.

        Args:
            name (str): The name of the importer.
            data_folder (str): The folder to store the downloaded data.
            kaggle_dataset_name (str): The name of the Kaggle dataset to download.
            destination_directory (str): The directory to move the CSV files to.
        """
        super().__init__(name)
        self.data_folder = data_folder
        self.kaggle_dataset_name = kaggle_dataset_name
        self.destination_directory = destination_directory

    def _download_data_kaggle(self):
        """
        Downloads the dataset from Kaggle.
        """
        if not os.path.exists(os.path.expanduser("~/.kaggle/kaggle.json")):
            raise Exception(
                "Kaggle API key not found. \
                Make sure to follow the instructions to set up your \
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

    def _move_csv_files_to_directory(self, destination_directory):
        """
        Moves all CSV files to the specified destination directory.

        Args:
            destination_directory (str): The directory to move the CSV files to.
        """
        os.makedirs(destination_directory, exist_ok=True)

        for root, dirs, files in os.walk(self.data_folder):
            for filename in files:
                if filename.endswith('.csv'):
                    filepath = os.path.join(root, filename)
                    new_filepath = os.path.join(destination_directory, filename)
                    shutil.move(filepath, new_filepath)

        print(
            f"All CSV files in {self.data_folder} have been moved to {destination_directory}")

    def _delete_except_directories(self, directory_to_keep):
        """
        Deletes all files and directories except for the specified directory.

        Args:
            directory_to_keep (str): The directory to keep and not delete.
        """
        directory_to_keep = os.path.abspath(directory_to_keep)

        for root, dirs, files in os.walk(self.data_folder, topdown=False):
            for filename in files:
                filepath = os.path.join(root, filename)
                if os.path.abspath(root) != directory_to_keep:
                    os.remove(filepath)

            for dirname in dirs:
                dirpath = os.path.join(root, dirname)
                if os.path.abspath(dirpath) != directory_to_keep:
                     shutil.rmtree(dirpath)

        print(
            f"All files and directories in {self.data_folder} except for {directory_to_keep} have been deleted")

    def execute(self):
        """
        Executes the import process.
        """
        print(f"Executing {self.name} step")
        self._download_data_kaggle()
        self._move_csv_files_to_directory(self.destination_directory)
        self._delete_except_directories(self.destination_directory)
        print(f"Finished executing {self.name} step")