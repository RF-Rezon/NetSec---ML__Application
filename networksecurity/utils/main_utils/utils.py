import sys,os
import yaml
import numpy as np
import pickle
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException


def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    


def write_yaml_file(file_path: str, content: object) -> None:
    try:
        directory = os.path.dirname(file_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file, default_flow_style=False, sort_keys=False)

    except Exception as e:
        raise NetworkSecurityException(e, sys)
    

def save_numpy_array_data(file_path: str, array: np.array):
    try:
        logging.info("Entered the save_numpy_array_data function")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
        logging.info("Exited the save_numpy_array_data function")

    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def save_object(file_path: str, obj: object) -> None:
    try:
        logging.info("Entered the save_object function")
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
        logging.info("Exited the save_object function")
    
    except Exception as e:
        raise NetworkSecurityException(e, sys)