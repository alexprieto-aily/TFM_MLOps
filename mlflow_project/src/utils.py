import os

def set_parent_directory_as_working_directory():
    """This function sets the parent directory as the working directory."""
    current_dir = os.path.abspath('.')
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)
