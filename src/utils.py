import os
import sys
import pickle

import pandas as pd
import numpy as np

from src.exception import CustomException

from sklearn.metrics import r2_score

def save_obj(file_path,obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file__obj:
            pickle.dump(obj, file__obj)
            
    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train, y_train, X_test, y_test, models):
    try:
        report = {}
        
        for model_name in models:  # Loop directly over the keys in the models dictionary
            model = models[model_name]  # Access each model by its name
            model.fit(X_train, y_train)
            
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)
            
            train_model_score = r2_score(y_train, y_train_pred)
            test_model_score = r2_score(y_test, y_test_pred)
            
            report[model_name] = test_model_score  # Store the test score with the model name as the key
        
        return report
    
    except Exception as e:
        raise CustomException(e, sys)