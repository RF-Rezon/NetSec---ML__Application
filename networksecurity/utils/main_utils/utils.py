import sys,os
import yaml
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