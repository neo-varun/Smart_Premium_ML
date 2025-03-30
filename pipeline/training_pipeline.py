import os
import sys
import pandas as pd
from components.data_ingestion import DataIngestion
from components.data_transformation import DataTransformation
from components.model_trainer import ModelTrainer
from components.model_evaluation import ModelEvaluation

def train_model():
    """
    Execute the full training pipeline:
    1. Data ingestion
    2. Data transformation
    3. Model training
    4. Model evaluation
    
    Returns:
        tuple: Containing the best model name and its score
    """
    try:
        # Data ingestion
        obj = DataIngestion()
        train_data_path, test_data_path = obj.data_ingestion()
        
        # Data transformation
        data_transformation = DataTransformation()
        train_arr, test_arr = data_transformation.initialize_data_transformation(train_data_path, test_data_path)
        
        # Model training
        model_trainer_obj = ModelTrainer()
        best_model_name, best_model_score = model_trainer_obj.initiate_model_training(train_arr, test_arr)
        
        # Model evaluation
        model_eval_obj = ModelEvaluation()
        model_eval_obj.initiate_model_evaluation(test_arr)
        
        return best_model_name, best_model_score
    
    except Exception as e:
        print(f"Exception occurred during training: {e}")
        raise e

# Execute training if file is run directly
if __name__ == "__main__":
    train_model()