import json
import os


def read_config(file_name, path=None):
    """
    Read config file of the project

    file_name: file name
    path : folder name

    """
    if path is None:
        file = file_name
    else:
        file = os.path.join(path, file_name)
    with open(file, "r") as f:
        config = json.load(f)

    return config