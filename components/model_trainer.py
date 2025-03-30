import os
import sys
import pickle
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score

class ModelTrainer:

    def __init__(self):

        self.trained_model_file_path = os.path.join('artifacts', 'models.pkl')

    def initiate_model_training(self, train_array, test_array):
    
        x_train, y_train, x_test, y_test = (
            train_array[:, :-1],
            train_array[:, -1],
            test_array[:, :-1],
            test_array[:, -1]
        )

        models = {
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(),
            'Random Forest': RandomForestRegressor(),
            'XGBoost': XGBRegressor()
        }

        report = {}
        model_objects = {}
        
        for model_name, model in models.items():
            model.fit(x_train, y_train)
            y_test_pred = model.predict(x_test)
            report[model_name] = r2_score(y_test, y_test_pred)
            model_objects[model_name] = model

        # Print model comparison in a more organized format
        print("\n" + "="*50)
        print("\033[1m{:<20} {:<15}\033[0m".format("MODEL", "R² SCORE"))
        print("="*50)
        for model_name, score in report.items():
            print("{:<20} {:<15.4f}".format(model_name, score))
        print("="*50 + "\n")

        best_model_name = max(report, key=report.get)
        best_model_score = report[best_model_name]
        best_model = model_objects[best_model_name]
        
        print(f"\033[1mBest Model: {best_model_name}\033[0m")
        print(f"R² Score: {best_model_score:.4f}")
        print("\n" + "="*50 + "\n")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.trained_model_file_path), exist_ok=True)
        
        # Save the best model to file
        with open(self.trained_model_file_path, 'wb') as f:
            pickle.dump(best_model, f)
            
        print(f"Model saved successfully to: {self.trained_model_file_path}")

        return best_model_name, best_model_score