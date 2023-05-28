import os

def set_parent_directory_as_working_directory() -> None:
    """
    Sets the parent directory as the working directory.
    """
    current_dir = os.path.abspath('.')
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)

def set_child_directory_as_working_directory(child_dir: str) -> None:
    """
    Sets the specified child directory as the working directory.

    Args:
        child_dir: The name of the child directory.
    """
    current_dir = os.path.abspath('.')
    child_dir = os.path.join(current_dir, child_dir)
    os.chdir(child_dir)

