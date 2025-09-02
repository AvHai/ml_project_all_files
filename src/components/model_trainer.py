# import os
# import sys
# from dataclasses import dataclass

# from catboost import CatBoostRegressor
# from sklearn.ensemble import (
#     AdaBoostRegressor,
#     GradientBoostingRegressor,
#     RandomForestRegressor,
# )
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import r2_score
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.tree import DecisionTreeRegressor
# from xgboost import XGBRegressor

# from src.exception import CustomException
# from src.logger import logging

# from src.utils import save_object,evaluate_models

# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path=os.path.join("artifacts","model.pkl")

# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config=ModelTrainerConfig()


#     def initiate_model_trainer(self,train_array,test_array):
#         try:
#             logging.info("Split training and test input data")
#             X_train,y_train,X_test,y_test=(
#                 train_array[:,:-1],
#                 train_array[:,-1],
#                 test_array[:,:-1],
#                 test_array[:,-1]
#             )
#             models = {
#                 "Random Forest": RandomForestRegressor(),
#                 "Decision Tree": DecisionTreeRegressor(),
#                 "Gradient Boosting": GradientBoostingRegressor(),
#                 "Linear Regression": LinearRegression(),
#                 "XGBRegressor": XGBRegressor(),
#                 "CatBoosting Regressor": CatBoostRegressor(verbose=False),
#                 "AdaBoost Regressor": AdaBoostRegressor(),
#             }
#             params={
#                 "Decision Tree": {
#                     'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
#                     # 'splitter':['best','random'],
#                     # 'max_features':['sqrt','log2'],
#                 },
#                 "Random Forest":{
#                     # 'criterion':['squared_error', 'friedman_mse', 'absolute_error', 'poisson'],
                 
#                     # 'max_features':['sqrt','log2',None],
#                     'n_estimators': [8,16,32,64,128,256]
#                 },
#                 "Gradient Boosting":{
#                     # 'loss':['squared_error', 'huber', 'absolute_error', 'quantile'],
#                     'learning_rate':[.1,.01,.05,.001],
#                     'subsample':[0.6,0.7,0.75,0.8,0.85,0.9],
#                     # 'criterion':['squared_error', 'friedman_mse'],
#                     # 'max_features':['auto','sqrt','log2'],
#                     'n_estimators': [8,16,32,64,128,256]
#                 },
#                 "Linear Regression":{},
#                 "XGBRegressor":{
#                     'learning_rate':[.1,.01,.05,.001],
#                     'n_estimators': [8,16,32,64,128,256]
#                 },
#                 "CatBoosting Regressor":{
#                     'depth': [6,8,10],
#                     'learning_rate': [0.01, 0.05, 0.1],
#                     'iterations': [30, 50, 100]
#                 },
#                 "AdaBoost Regressor":{
#                     'learning_rate':[.1,.01,0.5,.001],
#                     # 'loss':['linear','square','exponential'],
#                     'n_estimators': [8,16,32,64,128,256]
#                 }
                
#             }

#             model_report:dict=evaluate_models(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,
#                                              models=models,param=params)
            
#             ## To get best model score from dict
#             best_model_score = max(sorted(model_report.values()))

#             ## To get best model name from dict

#             best_model_name = list(model_report.keys())[
#                 list(model_report.values()).index(best_model_score)
#             ]
#             best_model = models[best_model_name]

#             if best_model_score<0.6:
#                 raise CustomException("No best model found")
#             logging.info(f"Best found model on both training and testing dataset")

#             save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=best_model
#             )

#             predicted=best_model.predict(X_test)

#             r2_square = r2_score(y_test, predicted)
#             return r2_square
            



            
#         except Exception as e:
#             raise CustomException(e,sys)

# import os
# import sys
# from dataclasses import dataclass
# import shap
# import numpy as np
# from xgboost import XGBRegressor
# from catboost import CatBoostRegressor
# from sklearn.metrics import r2_score

# from src.exception import CustomException
# from src.logger import logging
# from src.utils import save_object


# @dataclass
# class ModelTrainerConfig:
#     trained_model_file_path = os.path.join("artifacts", "model.pkl")
#     feature_importance_path = os.path.join("artifacts", "feature_importance.npy")


# class ModelTrainer:
#     def __init__(self):
#         self.model_trainer_config = ModelTrainerConfig()

#     def initiate_model_trainer(self, train_array, test_array):
#         try:
#             logging.info("Splitting training and test input data")
#             X_train, y_train, X_test, y_test = (
#                 train_array[:, :-1],
#                 train_array[:, -1],
#                 test_array[:, :-1],
#                 test_array[:, -1],
#             )

#             models = {
#                 "XGBoost": XGBRegressor(
#                     n_estimators=300,
#                     learning_rate=0.1,
#                     max_depth=6,
#                     subsample=0.8,
#                     colsample_bytree=0.8,
#                     early_stopping_rounds=20
#                 ),
#                 "CatBoost": CatBoostRegressor(
#                     iterations=300,
#                     learning_rate=0.1,
#                     depth=6,
#                     verbose=False,
#                     early_stopping_rounds=20
#                 )
#             }

#             best_model_name = None
#             best_score = -np.inf
#             best_model = None

#             for name, model in models.items():
#                 logging.info(f"Training {name}...")
#                 if name == "XGBoost":
#                     model.fit(X_train, y_train, eval_set=[(X_test, y_test)], verbose=False)
#                 else:
#                     model.fit(X_train, y_train, eval_set=(X_test, y_test), use_best_model=True)

#                 y_pred = model.predict(X_test)
#                 score = r2_score(y_test, y_pred)
#                 logging.info(f"{name} R2 Score: {score}")

#                 if score > best_score:
#                     best_score = score
#                     best_model_name = name
#                     best_model = model

#             if best_model is None or best_score < 0.6:
#                 raise CustomException("No suitable model found with acceptable performance")

#             logging.info(f"Best model: {best_model_name} with R2 Score: {best_score}")

#             save_object(
#                 file_path=self.model_trainer_config.trained_model_file_path,
#                 obj=best_model
#             )

#             # Feature importance with SHAP
#             logging.info("Calculating SHAP feature importance...")
#             explainer = shap.TreeExplainer(best_model)
#             shap_values = explainer.shap_values(X_test)
#             mean_abs_shap = np.abs(shap_values).mean(axis=0)
#             np.save(self.model_trainer_config.feature_importance_path, mean_abs_shap)
#             logging.info("Feature importance saved successfully.")

#             return best_score

#         except Exception as e:
#             raise CustomException(e, sys)

import os
import sys
from dataclasses import dataclass
import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score, confusion_matrix
import joblib

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")
    model_report_path: str = os.path.join("artifacts", "model_report.txt")

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        """
        Train classification models for flight delay prediction
        """
        try:
            logging.info("="*60)
            logging.info("STARTING MODEL TRAINING")
            logging.info("="*60)
            
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )
            
            logging.info(f"Training data: X_train {X_train.shape}, y_train {y_train.shape}")
            logging.info(f"Test data: X_test {X_test.shape}, y_test {y_test.shape}")
            
            # Check target distribution
            unique, counts = np.unique(y_train, return_counts=True)
            logging.info(f"Training target distribution: {dict(zip(unique, counts))}")
            
            unique_test, counts_test = np.unique(y_test, return_counts=True)
            logging.info(f"Test target distribution: {dict(zip(unique_test, counts_test))}")

            # Define classification models
            models = {
                "RandomForest": RandomForestClassifier(
                    n_estimators=100,
                    max_depth=10,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    random_state=42,
                    n_jobs=-1
                ),
                "XGBoost": XGBClassifier(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42,
                    eval_metric='logloss'
                ),
                "CatBoost": CatBoostClassifier(
                    iterations=100,
                    learning_rate=0.1,
                    depth=6,
                    verbose=False,
                    random_state=42
                )
            }

            best_model_name = None
            best_score = -np.inf
            best_model = None
            model_results = {}

            # Train and evaluate each model
            for name, model in models.items():
                logging.info(f"Training {name}...")
                
                try:
                    # Train the model
                    if name == "XGBoost":
                        model.fit(
                            X_train, y_train,
                            eval_set=[(X_test, y_test)],
                            verbose=False
                        )
                    elif name == "CatBoost":
                        model.fit(
                            X_train, y_train,
                            eval_set=(X_test, y_test),
                            use_best_model=True
                        )
                    else:
                        model.fit(X_train, y_train)
                    
                    # Make predictions
                    y_pred = model.predict(X_test)
                    y_pred_proba = model.predict_proba(X_test)[:, 1]
                    
                    # Calculate metrics
                    accuracy = accuracy_score(y_test, y_pred)
                    auc_score = roc_auc_score(y_test, y_pred_proba)
                    
                    # Store results
                    model_results[name] = {
                        'accuracy': accuracy,
                        'auc_score': auc_score,
                        'model': model
                    }
                    
                    logging.info(f"{name} Results:")
                    logging.info(f"  Accuracy: {accuracy:.4f}")
                    logging.info(f"  AUC Score: {auc_score:.4f}")
                    
                    # Use AUC score as the primary metric for model selection
                    if auc_score > best_score:
                        best_score = auc_score
                        best_model_name = name
                        best_model = model
                        
                except Exception as model_error:
                    logging.error(f"Error training {name}: {model_error}")
                    continue

            if best_model is None:
                raise CustomException("No model could be trained successfully")
                
            if best_score < 0.60:  # Minimum acceptable AUC score
                logging.warning(f"Best model AUC score ({best_score:.4f}) is below threshold (0.60)")
            
            logging.info(f"Best model: {best_model_name} with AUC Score: {best_score:.4f}")

            # Generate detailed report for best model
            y_pred_best = best_model.predict(X_test)
            y_pred_proba_best = best_model.predict_proba(X_test)[:, 1]
            
            accuracy_best = accuracy_score(y_test, y_pred_best)
            auc_best = roc_auc_score(y_test, y_pred_proba_best)
            
            # Classification report
            class_report = classification_report(y_test, y_pred_best)
            conf_matrix = confusion_matrix(y_test, y_pred_best)
            
            # Save model
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
            )
            
            # Save detailed report
            with open(self.model_trainer_config.model_report_path, 'w') as f:
                f.write("="*60 + "\n")
                f.write("FLIGHT DELAY PREDICTION MODEL REPORT\n")
                f.write("="*60 + "\n\n")
                
                f.write(f"Best Model: {best_model_name}\n")
                f.write(f"Training Data Size: {X_train.shape[0]:,} samples\n")
                f.write(f"Test Data Size: {X_test.shape[0]:,} samples\n")
                f.write(f"Number of Features: {X_train.shape[1]}\n\n")
                
                f.write("PERFORMANCE METRICS:\n")
                f.write(f"Accuracy: {accuracy_best:.4f}\n")
                f.write(f"AUC Score: {auc_best:.4f}\n\n")
                
                f.write("CONFUSION MATRIX:\n")
                f.write(f"{conf_matrix}\n\n")
                
                f.write("CLASSIFICATION REPORT:\n")
                f.write(class_report + "\n\n")
                
                f.write("ALL MODEL RESULTS:\n")
                for model_name, results in model_results.items():
                    f.write(f"{model_name}:\n")
                    f.write(f"  Accuracy: {results['accuracy']:.4f}\n")
                    f.write(f"  AUC Score: {results['auc_score']:.4f}\n\n")

            # Feature importance (if available)
            try:
                if hasattr(best_model, 'feature_importances_'):
                    feature_importance = best_model.feature_importances_
                    importance_path = os.path.join("artifacts", "feature_importance.npy")
                    np.save(importance_path, feature_importance)
                    logging.info(f"Feature importance saved to {importance_path}")
                    
                    # Log top 10 most important features
                    top_indices = np.argsort(feature_importance)[::-1][:10]
                    logging.info("Top 10 Feature Importances:")
                    for i, idx in enumerate(top_indices, 1):
                        logging.info(f"  {i}. Feature {idx}: {feature_importance[idx]:.4f}")
                        
            except Exception as importance_error:
                logging.warning(f"Could not extract feature importance: {importance_error}")

            logging.info("="*60)
            logging.info("MODEL TRAINING COMPLETED SUCCESSFULLY")
            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Best AUC Score: {best_score:.4f}")
            logging.info(f"Model saved to: {self.model_trainer_config.trained_model_file_path}")
            logging.info(f"Report saved to: {self.model_trainer_config.model_report_path}")
            logging.info("="*60)
            
            return best_score

        except Exception as e:
            logging.error(f"Error in model training: {str(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            raise CustomException(e, sys)