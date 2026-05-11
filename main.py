from networksecurity.components.data_ingestion import DataIngestion
from networksecurity.components.data_validation import DataValidation
from networksecurity.components.data_transformation import DataTransformation
from networksecurity.components.model_trainer import ModelTrainer
from networksecurity.exception.exception import NetworkSecurityException
from networksecurity.logging.logger import logging

from networksecurity.entity.config_entity import DataIngestionConfig, DataValidationConfig, DataTransformationConfig, ModelTrainerConfig
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
        dataTransformationConfig = DataTransformationConfig(trainingPipelineConfig)
        dataTransformation = DataTransformation(data_validation_artifact, dataTransformationConfig)
        data_transformation_artifact = dataTransformation.initiate_data_transformation()
        
        Model_TrainerConfig = ModelTrainerConfig(trainingPipelineConfig)
        modelTrainer = ModelTrainer(Model_TrainerConfig, data_transformation_artifact)
        model_trainer_artifact = modelTrainer.initiate_model_trainer()
        
        print(model_trainer_artifact)

    except Exception as e:
        raise NetworkSecurityException(e,sys)