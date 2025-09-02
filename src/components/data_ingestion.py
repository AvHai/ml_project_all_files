import os
import sys
from src.exception import CustomException
from src.logger import logging
import pandas as pd

from sklearn.model_selection import train_test_split
from dataclasses import dataclass

from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer

@dataclass
class DataIngestionConfig:
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")
    raw_data_path: str = os.path.join('artifacts', "data.csv")

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Entered the data ingestion method")
        try:
            # Read the dataset
            df = pd.read_csv('notebook/data/flight_delay_2015_cleaned.csv', 
                           low_memory=False, 
                           dtype={'TAIL_NUMBER': str, 'CANCELLATION_REASON': str})
            logging.info('Read the dataset as dataframe')
            logging.info(f"Dataset shape: {df.shape}")

            # Create artifacts directory
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)

            # Save raw data
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)

            logging.info("Train test split initiated")
            
            # Stratified split based on delay
            # Create temporary delay indicator for stratification
            delay_indicator = (df['ARRIVAL_DELAY'] >= 15).astype(int)
            
            train_set, test_set = train_test_split(
                df, 
                test_size=0.2, 
                random_state=42,
                stratify=delay_indicator
            )

            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)

            logging.info("Data ingestion completed")
            logging.info(f"Train set: {train_set.shape}, Test set: {test_set.shape}")

            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
            
        except Exception as e:
            logging.error(f"Error in data ingestion: {e}")
            raise CustomException(e, sys)

def main():
    """Complete pipeline execution function"""
    try:
        logging.info("="*60)
        logging.info("STARTING COMPLETE ML PIPELINE")
        logging.info("="*60)
        
        # Step 1: Data Ingestion
        logging.info("STEP 1: DATA INGESTION")
        logging.info("-" * 30)
        obj = DataIngestion()
        train_data_path, test_data_path = obj.initiate_data_ingestion()
        
        # Step 2: Data Transformation
        logging.info("STEP 2: DATA TRANSFORMATION")
        logging.info("-" * 30)
        data_transformation = DataTransformation()
        train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(
            train_data_path, test_data_path
        )
        
        # Step 3: Model Training
        logging.info("STEP 3: MODEL TRAINING")
        logging.info("-" * 30)
        model_trainer = ModelTrainer()
        model_score = model_trainer.initiate_model_trainer(train_arr, test_arr)
        
        logging.info("="*60)
        logging.info("COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
        logging.info("="*60)
        logging.info("FINAL RESULTS:")
        logging.info(f"  • Train data shape: {train_arr.shape}")
        logging.info(f"  • Test data shape: {test_arr.shape}")
        logging.info(f"  • Model AUC Score: {model_score:.4f}")
        logging.info(f"  • Preprocessor saved: {preprocessor_path}")
        logging.info(f"  • Model saved: artifacts/model.pkl")
        logging.info(f"  • Model report: artifacts/model_report.txt")
        logging.info("="*60)
        
        return {
            'train_data_shape': train_arr.shape,
            'test_data_shape': test_arr.shape,
            'model_score': model_score,
            'preprocessor_path': preprocessor_path,
            'model_path': 'artifacts/model.pkl'
        }
        
    except Exception as e:
        logging.error(f"Pipeline execution failed: {e}")
        print(f"Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    results = main()
    print(f"\n🎉 Pipeline completed successfully!")
    print(f"📊 Model AUC Score: {results['model_score']:.4f}")
    print(f"📁 Check 'artifacts' folder for saved models and reports")