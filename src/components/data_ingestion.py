import os
import sys

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataTransformationConfig
from src.components.model_trainer import ModelTrainer
from src.components.model_trainer import ModelTrainerConfig
@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join("artifacts", 'train.csv')
    test_data_path: str = os.path.join("artifacts", 'test.csv')
    raw_data_path: str = os.path.join("artifacts", 'raw.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method or component")
        try:
            # Load the data
            df = pd.read_csv("notebook/data/stud.csv")  # Use forward slashes or double backslashes
            logging.info("Read the dataset as DataFrame")
            
            # Create directory if not exists and save raw data
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            
            # Split the data into training and testing sets
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2, random_state=42)
            
            # Save the training and testing sets
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            
            logging.info("Data Ingestion completed")
            
            return self.ingestion_config.train_data_path, self.ingestion_config.test_data_path
        
        except Exception as e:
            logging.error("Data Ingestion failed")
            raise CustomException(e, sys)
        
if __name__ == "__main__":
    obj = DataIngestion()
    train_data,test_data = obj.initiate_data_ingestion()
    
    data_transformation = DataTransformation()
    train_arr , test_arr ,_ = data_transformation.initiate_data_transformation(train_data, test_data)
    
    model_trainer = ModelTrainer()
    print(model_trainer.initiate_model_training(train_arr, test_arr))
