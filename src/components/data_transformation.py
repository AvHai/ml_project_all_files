import sys
from dataclasses import dataclass
import os
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler, LabelEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
import sklearn

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        logging.info(f"Sklearn version: {sklearn.__version__}")

    def get_part_of_day(self, x):
        try:
            if pd.isna(x):
                return 'Unknown'
            x = int(float(x))
            if 500 <= x < 1200:
                return 'Morning'
            elif 1200 <= x < 1700:
                return 'Afternoon'
            elif 1700 <= x < 2100:
                return 'Evening'
            else:
                return 'Night'
        except (ValueError, TypeError):
            return 'Unknown'

    def get_data_transformer_object(self, df: pd.DataFrame, target_col="DELAYED"):
        """
        Creates the transformation pipeline using only built-in sklearn transformers
        """
        try:
            logging.info("Starting get_data_transformer_object")
            logging.info(f"Input dataframe shape: {df.shape}")
            
            # Create a copy to avoid modifying original
            df_work = df.copy()
            
            # 1. Create target BEFORE dropping columns if it doesn't exist
            if target_col not in df_work.columns and 'ARRIVAL_DELAY' in df_work.columns:
                df_work[target_col] = (df_work['ARRIVAL_DELAY'] >= 15).astype(int)
                logging.info(f"Created target column: {target_col}")
            
            # 2. Drop leakage columns
            leakage_cols = [
                'DEPARTURE_TIME', 'DEPARTURE_DELAY', 'WHEELS_OFF', 'WHEELS_ON',
                'TAXI_IN', 'TAXI_OUT', 'ARRIVAL_TIME', 'ARRIVAL_DELAY',
                'AIR_SYSTEM_DELAY', 'SECURITY_DELAY', 'AIRLINE_DELAY',
                'LATE_AIRCRAFT_DELAY', 'WEATHER_DELAY', 'DIVERTED', 'CANCELLED',
                'CANCELLATION_REASON'
            ]

            cols_to_drop = [col for col in leakage_cols if col in df_work.columns]
            if cols_to_drop:
                df_work = df_work.drop(columns=cols_to_drop)
                logging.info(f"Dropped {len(cols_to_drop)} leakage columns")

            # 3. Handle missing values for critical columns
            critical_cols = ['YEAR', 'MONTH', 'DAY', 'AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT']
            existing_critical = [col for col in critical_cols if col in df_work.columns]
            if existing_critical:
                before_drop = len(df_work)
                df_work = df_work.dropna(subset=existing_critical)
                logging.info(f"Dropped {before_drop - len(df_work)} rows with missing critical values")

            # 4. Handle outliers: Winsorize DISTANCE
            if 'DISTANCE' in df_work.columns:
                q_low, q_high = df_work['DISTANCE'].quantile([0.01, 0.99])
                df_work['DISTANCE'] = np.clip(df_work['DISTANCE'], q_low, q_high)

            # 5. Feature engineering
            # Create date features
            if all(col in df_work.columns for col in ['YEAR', 'MONTH', 'DAY']):
                df_work['DAY_OF_WEEK'] = df_work.get('DAY_OF_WEEK', 
                    pd.to_datetime(df_work[['YEAR', 'MONTH', 'DAY']], errors='coerce').dt.dayofweek)
                df_work['IS_WEEKEND'] = (df_work['DAY_OF_WEEK'] >= 5).astype(int)
                
                # Create cyclical features for month
                df_work['MONTH_SIN'] = np.sin(2 * np.pi * df_work['MONTH'] / 12)
                df_work['MONTH_COS'] = np.cos(2 * np.pi * df_work['MONTH'] / 12)

            # Create part of day feature
            if 'SCHEDULED_DEPARTURE' in df_work.columns:
                df_work['PART_OF_DAY'] = df_work['SCHEDULED_DEPARTURE'].apply(self.get_part_of_day)
                logging.info("Created PART_OF_DAY feature")

            # Create efficiency features
            if all(col in df_work.columns for col in ['DISTANCE', 'SCHEDULED_TIME']):
                df_work['DISTANCE_PER_MINUTE'] = df_work['DISTANCE'] / (df_work['SCHEDULED_TIME'] + 1)

            # 6. Define columns for pipeline - SIMPLIFIED APPROACH
            # Use OrdinalEncoder for ALL categorical columns to avoid high dimensionality
            categorical_candidates = ['AIRLINE', 'ORIGIN_AIRPORT', 'DESTINATION_AIRPORT', 'PART_OF_DAY']
            categorical_cols = [col for col in categorical_candidates if col in df_work.columns]
            
            # Get numerical columns (excluding target)
            numerical_cols = [col for col in df_work.select_dtypes(include=[np.number]).columns 
                            if col != target_col]

            logging.info(f"Categorical columns: {categorical_cols}")
            logging.info(f"Numerical columns: {len(numerical_cols)} columns")

            # 7. Create simple pipelines using only built-in transformers
            transformers = []
            
            # Numerical pipeline
            if numerical_cols:
                num_pipeline = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy="median")),
                    ('scaler', RobustScaler())
                ])
                transformers.append(('num', num_pipeline, numerical_cols))
                logging.info(f"Created numerical pipeline for {len(numerical_cols)} columns")

            # Categorical pipeline using OrdinalEncoder (handles high cardinality gracefully)
            if categorical_cols:
                cat_pipeline = Pipeline(steps=[
                    ('imputer', SimpleImputer(strategy="most_frequent")),
                    ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
                ])
                transformers.append(('cat', cat_pipeline, categorical_cols))
                logging.info(f"Created categorical pipeline for {len(categorical_cols)} columns")

            if not transformers:
                raise ValueError("No valid transformers created. Check your data columns.")

            preprocessor = ColumnTransformer(
                transformers=transformers,
                remainder='drop'
            )

            logging.info(f"Created preprocessor with {len(transformers)} transformer(s)")
            
            return df_work, preprocessor

        except Exception as e:
            logging.error(f"Error in get_data_transformer_object: {str(e)}")
            import traceback
            logging.error(f"Traceback: {traceback.format_exc()}")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path: str, test_path: str):
        """
        Main data transformation method
        """
        try:
            logging.info("="*60)
            logging.info("STARTING SIMPLE DATA TRANSFORMATION")
            logging.info("="*60)
            
            # Read data
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)
            logging.info(f"Read train data: {train_df.shape}")
            logging.info(f"Read test data: {test_df.shape}")

            target_column_name = "DELAYED"

            # Process training data and create preprocessor
            logging.info("Processing training data...")
            train_df_processed, preprocessing_obj = self.get_data_transformer_object(train_df, target_col=target_column_name)
            
            # Process test data with same logic
            logging.info("Processing test data...")
            test_df_processed, _ = self.get_data_transformer_object(test_df, target_col=target_column_name)

            # Verify target column exists
            if target_column_name not in train_df_processed.columns:
                raise ValueError(f"Target column {target_column_name} not found in training data")
            if target_column_name not in test_df_processed.columns:
                raise ValueError(f"Target column {target_column_name} not found in test data")

            # Separate features and target
            input_feature_train_df = train_df_processed.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df_processed[target_column_name]

            input_feature_test_df = test_df_processed.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df_processed[target_column_name]

            logging.info(f"Training features: {input_feature_train_df.shape}")
            logging.info(f"Test features: {input_feature_test_df.shape}")
            
            # Log feature column names for debugging
            logging.info(f"Feature columns: {list(input_feature_train_df.columns)}")

            # Apply preprocessing
            logging.info("Fitting preprocessor on training data...")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            
            logging.info("Transforming test data...")
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            logging.info(f"Transformed training features: {input_feature_train_arr.shape}")
            logging.info(f"Transformed test features: {input_feature_test_arr.shape}")

            # Combine features and targets
            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            # Save preprocessing object
            logging.info("Saving preprocessor...")
            os.makedirs(os.path.dirname(self.data_transformation_config.preprocessor_obj_file_path), exist_ok=True)
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            logging.info("="*60)
            logging.info("DATA TRANSFORMATION COMPLETED SUCCESSFULLY")
            logging.info(f"Final train shape: {train_arr.shape}")
            logging.info(f"Final test shape: {test_arr.shape}")
            logging.info(f"Preprocessor saved: {self.data_transformation_config.preprocessor_obj_file_path}")
            logging.info("="*60)

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            logging.error(f"Error in data transformation: {str(e)}")
            import traceback
            logging.error(f"Full traceback: {traceback.format_exc()}")
            raise CustomException(e, sys)