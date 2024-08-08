import sys
from dataclasses import dataclass
import os
import pickle

import pandas as pd 
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join("artifacts", 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        
    def get_data_transformer_obj(self):
        logging.info("Creating data transformer object")

        try:
            numerical_columns = ["writing_score", "reading_score"]
            categorical_columns = [
                "gender",
                "race_ethnicity",
                "parental_level_of_education",
                "lunch",
                "test_preparation_course"
            ]
            
            # Pipeline for numerical features
            num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="median")),
                    ('scaler', StandardScaler())
                ]
            )
            
            # Pipeline for categorical features
            cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ('one_hot_encoder', OneHotEncoder()),
                    ('scaler', StandardScaler(with_mean=False))  # with_mean=False for sparse data
                ]
            )
            
            logging.info(f"Numerical columns: {numerical_columns}")
            logging.info(f"Categorical columns: {categorical_columns}")
            
            # Combine both pipelines
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )
            
            logging.info("Data transformer object created successfully")
            return preprocessor

        except Exception as e:
            logging.error("Error in creating data transformer object")
            raise CustomException(e, sys)
    
    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            
            logging.info("Training and testing datasets loaded successfully")
            
            target_column_name = "math_score"
            
            # Features and target separation
            input_features_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]
            
            input_features_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]
            
            logging.info("Applying preprocessor object on datasets")
            preprocessing_obj = self.get_data_transformer_obj()
            
            input_features_train_arr = preprocessing_obj.fit_transform(input_features_train_df)
            input_features_test_arr = preprocessing_obj.transform(input_features_test_df)
            
            # Combine the transformed features with the target
            train_arr = np.c_[input_features_train_arr, target_feature_train_df.to_numpy()]
            test_arr = np.c_[input_features_test_arr, target_feature_test_df.to_numpy()]
            
            logging.info("Saving preprocessor object")
            save_obj(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            
            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except KeyError as e:
            logging.error(f"Column {str(e)} not found in the dataset")
            raise CustomException(e, sys)

        except Exception as e:
            logging.error("Error in data transformation process")
            raise CustomException(e, sys)
