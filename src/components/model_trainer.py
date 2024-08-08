import os
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
from src.utils import save_obj
from src.utils import evaluate_model

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", 'model.pkl')
    
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()
        
    def initiate_model_training(self, train_array, test_array):
        try:
            logging.info("Split train and test data into features and target")
            X_train, y_train , X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            models = {
                "Linear Regression": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "AdaBoost": AdaBoostRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "XGBoost": XGBRegressor(),
                "KNN": KNeighborsRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False)
            }
            
            params = {
                "Linear Regression": {},
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"],},
                "Random Forest": {"n_estimators": [50,100,150,200,250]},
                "AdaBoost": {
                    "learning_rate": [0.01,0.05,0.1,0.3,0.5],
                    "n_estimators": [50,100,150,200,250]},
                "Gradient Boosting": {
                    "n_estimators": [50,100,150,200,250],
                    "learning_rate": [0.01,0.05,0.1,0.3,0.5],
                    "subsample": [0.5,0.7,1.0]
                },
                "XGBoost": {
                    "n_estimators": [50,100,150,200,250],
                    "learning_rate": [0.01,0.05,0.1,0.3,0.5],
                    },
                "KNN": {"n_neighbors": [3,5,7,9,11]},
                "CatBoost": {
                    "iterations": [50,100,150,200,250],
                    "learning_rate": [0.01,0.05,0.1,0.3,0.5],
                    "depth": [3,5,7,9,11]
                }
            }
        
            model_report: dict = evaluate_model(X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, models=models, params=params)
            
            
            best_model_score = max(sorted(model_report.values()))
            
            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            
            best_model = models[best_model_name]
            
            if best_model_score < 0.5:
                logging.error("Model score is less than 0.5")
                raise CustomException("No best model found")
            logging.info("Best model found on both train and test data")
            
            save_obj(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            predicted = best_model.predict(X_test)
            r2 = r2_score(y_test, predicted)
            
            return r2
        except Exception as e:
            logging.error("Model Training failed")
            raise CustomException(e, sys)
