import os
import pickle
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn

class PredictPipeline:
    
    def __init__(self):
        self.preprocessor_path = os.path.join('artifacts', 'preprocessor.pkl')
        self.model_path = os.path.join('artifacts', 'models.pkl')
        # Set MLflow tracking URI to connect to the local server
        mlflow.set_tracking_uri('http://localhost:5000')
        print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
        
    def load_preprocessor(self, run_id=None):
        """Load the preprocessor from local storage or MLflow."""
        if run_id:
            try:
                print(f"Attempting to load preprocessor from MLflow run: {run_id}")
                # Try loading from MLflow artifacts
                client = mlflow.tracking.MlflowClient()
                artifact_path = client.download_artifacts(run_id, "preprocessor.pkl")
                with open(artifact_path, 'rb') as file:
                    preprocessor = pickle.load(file)
                print(f"Successfully loaded preprocessor from MLflow artifacts")
                return preprocessor
            except Exception as e:
                print(f"Failed to load preprocessor from MLflow. Using local file. Error: {e}")
                
        # Use local file as fallback
        if not os.path.exists(self.preprocessor_path):
            raise FileNotFoundError(f"Preprocessor file not found at {self.preprocessor_path}. Please train the model first.")
            
        with open(self.preprocessor_path, 'rb') as file:
            preprocessor = pickle.load(file)
        return preprocessor

    def predict(self, features, run_id=None):
        """Make predictions using the trained model and preprocessor."""
        
        # Handle date transformation separately before passing to preprocessor
        if 'Policy Start Date' in features.columns:
            features['Policy Start Date'] = pd.to_datetime(features['Policy Start Date'], format='%d-%m-%Y', errors='coerce')

        # Load preprocessor
        preprocessor = self.load_preprocessor(run_id)

        # Load model from MLflow or local storage
        if run_id:
            try:
                print(f"Attempting to load model from MLflow run: {run_id}")
                model_uri = f"runs:/{run_id}/model"
                model = mlflow.sklearn.load_model(model_uri)
                model_type = type(model).__name__
                print(f"Successfully loaded {model_type} model from MLflow Run ID: {run_id}")
                print("Using MLflow model for prediction")
            except Exception as e:
                print(f"Error loading from MLflow: {e}")
                # If MLflow fails, use local model
                if not os.path.exists(self.model_path):
                    raise FileNotFoundError(f"Model file not found at {self.model_path} and MLflow loading failed.")
                with open(self.model_path, 'rb') as file:
                    model = pickle.load(file)
                print("Fallback to local model due to MLflow error")
        else:
            # Load model from local storage
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found at {self.model_path}. Please train the model first.")
            
            with open(self.model_path, 'rb') as file:
                model = pickle.load(file)
            print("Using local model for prediction")

        # Debug feature count
        print(f"Input features shape before preprocessing: {features.shape}")

        # Transform features using preprocessor
        transformed_features = preprocessor.transform(features)
        
        # Handle feature names to match model expectations
        if hasattr(model, 'feature_names_in_'):
            # If model expects generic feature names like 'feature_0', 'feature_1', etc.
            expected_feature_names = model.feature_names_in_
            if all(fname.startswith('feature_') for fname in expected_feature_names):
                # Create a DataFrame with generic feature names
                transformed_features_df = pd.DataFrame(
                    transformed_features, 
                    columns=[f'feature_{i}' for i in range(transformed_features.shape[1])]
                )
                print("Using generic feature names to match model expectations")
            else:
                # Use the preprocessor's feature names if they don't follow generic pattern
                transformed_features_df = pd.DataFrame(
                    transformed_features, 
                    columns=preprocessor.get_feature_names_out()
                )
        else:
            # If model doesn't have feature_names_in_ attribute
            transformed_features_df = pd.DataFrame(
                transformed_features, 
                columns=preprocessor.get_feature_names_out()
            )

        print(f"Transformed features shape: {transformed_features_df.shape}")

        # Check if model has feature_names_in_ attribute (scikit-learn >= 0.24)
        if hasattr(model, 'feature_names_in_'):
            # Ensure feature names match training data
            missing_features = set(model.feature_names_in_) - set(transformed_features_df.columns)
            if missing_features:
                print(f"Warning: Feature name mismatch. Model expects: {sorted(model.feature_names_in_)}")
                print(f"Got features: {sorted(transformed_features_df.columns)}")
                raise ValueError(f"Feature mismatch! Model expects {len(model.feature_names_in_)} features, but got {len(transformed_features_df.columns)}.\nMissing features: {missing_features}")

        # Make predictions
        predictions = model.predict(transformed_features_df)
        return predictions

class CustomData:
    
    def __init__(self, age, annual_income, num_dependents, health_score, previous_claims, 
                 vehicle_age, credit_score, insurance_duration, policy_start_date, gender, 
                 marital_status, education_level, occupation, location, policy_type, 
                 customer_feedback, smoking_status, exercise_frequency, property_type):
        
        # Numerical features
        self.age = age
        self.annual_income = annual_income
        self.num_dependents = num_dependents
        self.health_score = health_score
        self.previous_claims = previous_claims
        self.vehicle_age = vehicle_age
        self.credit_score = credit_score
        self.insurance_duration = insurance_duration
        
        # Date feature (convert to days since today)
        self.policy_start_date = policy_start_date

        # Categorical features
        self.gender = gender
        self.marital_status = marital_status
        self.education_level = education_level
        self.occupation = occupation
        self.location = location
        self.policy_type = policy_type
        self.customer_feedback = customer_feedback
        self.smoking_status = smoking_status
        self.exercise_frequency = exercise_frequency
        self.property_type = property_type

    def get_data_as_dataframe(self):
        
        custom_data_dict = {
            'Age': [self.age],
            'Annual Income': [self.annual_income],
            'Number of Dependents': [self.num_dependents],
            'Health Score': [self.health_score],
            'Previous Claims': [self.previous_claims],
            'Vehicle Age': [self.vehicle_age],
            'Credit Score': [self.credit_score],
            'Insurance Duration': [self.insurance_duration],
            'Policy Start Date': [self.policy_start_date],
            'Gender': [self.gender],
            'Marital Status': [self.marital_status],
            'Education Level': [self.education_level],
            'Occupation': [self.occupation],
            'Location': [self.location],
            'Policy Type': [self.policy_type],
            'Customer Feedback': [self.customer_feedback],
            'Smoking Status': [self.smoking_status],
            'Exercise Frequency': [self.exercise_frequency],
            'Property Type': [self.property_type]
        }

        df = pd.DataFrame(custom_data_dict)
        return df