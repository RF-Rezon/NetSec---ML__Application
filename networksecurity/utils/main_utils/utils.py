import sys,os
import yaml
import numpy as np
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import RandomizedSearchCV
from networksecurity.logging.logger import logging
from networksecurity.exception.exception import NetworkSecurityException


def write_yaml_file(file_path: str, content: object) -> None:
    try:
        directory = os.path.dirname(file_path)

        if directory:
            os.makedirs(directory, exist_ok=True)

        with open(file_path, "w") as file:
            yaml.dump(content, file, default_flow_style=False, sort_keys=False)

    except Exception as e:
        raise NetworkSecurityException(e, sys)
    

def read_yaml_file(file_path: str) -> dict:
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    


def save_numpy_array_data(file_path: str, array: np.array):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            np.save(file_obj, array)
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    

def load_numpy_array_data(file_path: str) -> np.array:
    try:
        with open(file_path, "rb") as file_obj:
            return np.load(file_obj)
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    

    
def save_object(file_path: str, obj: object) -> None:
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)
    
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
def load_object(file_path: str) -> object:
    try:
        if not os.path.exists(file_path):
            raise Exception(f"The file: {file_path} is not exists")
        
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)
        
    except Exception as e:
        raise NetworkSecurityException(e, sys)
    
    
def evaluate_models(X_train, y_train, X_test, y_test, models, param):
    try:
        report = {}
        
        # Render এ RENDER environment variable automatically থাকে
        is_render = os.environ.get("RENDER") is not None

        for name, model in models.items():
            logging.info(f"🚀 Training started: {name}")
            para = param[name]

            if is_render:
                # Render: fast training, no tuning
                model.fit(X_train, y_train)
                best_model = model
            else:
                # Localhost: full hyperparameter tuning
                gs = RandomizedSearchCV(
                    estimator=model,
                    param_distributions=para,
                    cv=3,
                    verbose=1,
                    n_iter=10,
                    n_jobs=-1
                )
                gs.fit(X_train, y_train)
                best_model = gs.best_estimator_

            y_train_pred = best_model.predict(X_train)
            y_test_pred = best_model.predict(X_test)

            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            logging.info(f"{name} | Train: {train_score:.4f} | Test: {test_score:.4f}")
            report[name] = test_score

        return report

    except Exception as e:
        raise NetworkSecurityException(e, sys)