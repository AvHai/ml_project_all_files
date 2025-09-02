import os
import sys
import numpy as np 
import pandas as pd
import dill
import pickle
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV
from src.logger import logging
from src.exception import CustomException

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
# def evaluate_models(X_train, y_train,X_test,y_test,models,param):
#     try:
#         report = {}

#         for i in range(len(list(models))):
#             model = list(models.values())[i]
#             para=param[list(models.keys())[i]]

#             gs = GridSearchCV(model,para,cv=3)
#             gs.fit(X_train,y_train)

#             model.set_params(**gs.best_params_)
#             model.fit(X_train,y_train)

#             #model.fit(X_train, y_train)  # Train model

#             y_train_pred = model.predict(X_train)

#             y_test_pred = model.predict(X_test)

#             train_model_score = r2_score(y_train, y_train_pred)

#             test_model_score = r2_score(y_test, y_test_pred)

#             report[list(models.keys())[i]] = test_model_score

#         return report

#     except Exception as e:
#         raise CustomException(e, sys)
    
def evaluate_models(X_train, y_train, X_test, y_test, models):
    """
    Evaluate multiple models quickly without hyperparameter tuning.
    Uses R² score for comparison.
    """
    try:
        report = {}

        for name, model in models.items():
            logging.info(f"Training and evaluating {name}...")

            # Fit the model (with early stopping if available)
            if hasattr(model, "fit"):
                if name == "XGBoost":
                    model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
                elif name == "CatBoost":
                    model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True, verbose=False)
                else:
                    model.fit(X_train, y_train)
            else:
                raise CustomException(f"Model {name} does not have a .fit() method")

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # R² scores
            train_score = r2_score(y_train, y_train_pred)
            test_score = r2_score(y_test, y_test_pred)

            logging.info(f"{name} - Train R²: {train_score}, Test R²: {test_score}")
            report[name] = test_score

        return report

    except Exception as e:
        raise CustomException(e, sys)
    
    
def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return pickle.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)
