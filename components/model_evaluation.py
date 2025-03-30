import numpy as np
import pickle
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import os

class ModelEvaluation:
    def __init__(self, model_path="artifacts/models.pkl", preprocessor_path="artifacts/preprocessor.pkl"):
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        
    def eval_metrics(self, actual, pred):
        rmse = np.sqrt(mean_squared_error(actual, pred))
        mae = mean_absolute_error(actual, pred)
        r2 = r2_score(actual, pred)
        return rmse, mae, r2

    def initiate_model_evaluation(self, test_array):
        x_test, y_test = test_array[:, :-1], test_array[:, -1]

        # Load the trained model
        with open(self.model_path, 'rb') as file:
            model = pickle.load(file)
            
        # Check if preprocessor exists
        if not os.path.exists(self.preprocessor_path):
            print(f"Warning: Preprocessor not found at {self.preprocessor_path}")

        # Make predictions using the model
        prediction = model.predict(x_test)
        rmse, mae, r2 = self.eval_metrics(y_test, prediction)

        # Print evaluation results in a formatted table
        print("\n" + "="*50)
        print("\033[1mMODEL EVALUATION METRICS\033[0m")
        print("="*50)
        print(f"{'RMSE:':<15} {rmse:.4f}")
        print(f"{'MAE:':<15} {mae:.4f}")
        print(f"{'R² Score:':<15} {r2:.4f}")
        print("="*50)
        
        print("\nModel evaluation completed")
        print("="*50 + "\n")
        
        return rmse, mae, r2