from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import DataIngestionConfig, DataValidationConfig
from networksecurity.entity.config_entity import TrainingPipelineConfig

import sys

if __name__=='__main__':
    try:
        trainingPipelineConfig = TrainingPipelineConfig()
        dataIngestionConfig = DataIngestionConfig(trainingPipelineConfig)
        dataIngestion = DataIngestion(dataIngestionConfig)

        logging.info("Initiate the data ingestion.")

        dataIngestionartifact = dataIngestion.initiate_data_ingestion()

        dataValidationConfig = DataValidationConfig(trainingPipelineConfig)
        dataValidation = DataValidation(dataIngestionartifact,dataValidationConfig)

        logging.info("Initiate the data validation.")

        
        data_validation_artifact = dataValidation.initiate_data_validation()
        logging.info("data Validation Completed")
        print(data_validation_artifact)
        

    except Exception as e:
        raise NetworkSecurityException(e,sys)