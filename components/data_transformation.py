import pandas as pd
import numpy as np
import os
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import pickle

class DataTransformation:

    def __init__(self):
        self.preprocessor_obj_file_path = os.path.join('artifacts', 'preprocessor.pkl')

    def initiate_data_transformation(self):

        categorical_cols=['Gender','Marital Status','Education Level','Occupation','Location','Policy Type','Customer Feedback',
                          'Smoking Status','Exercise Frequency','Property Type',]
        numerical_cols=['Age','Annual Income','Number of Dependents','Health Score','Previous Claims',
                        'Vehicle Age','Credit Score','Insurance Duration']
        date_col=['Policy Start Date']

        gender_categories=['Female', 'Male']
        martial_categories=['Married', 'Divorced', 'Single']
        education_categories=["Bachelor's", "Master's", 'High School', 'PhD']
        occupation_categories=['Self-Employed', 'Employed', 'Unemployed']
        location_categories=['Urban', 'Rural', 'Suburban']
        policy_categories=['Premium', 'Comprehensive', 'Basic']
        feedback_categories=['Poor', 'Average', 'Good']
        smoking_categories=['No', 'Yes']
        exercise_categories=['Weekly', 'Monthly', 'Daily', 'Rarely']
        property_categories=['House', 'Apartment', 'Condo']

        num_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='median')),
                    ('scalar', StandardScaler())
                ]
            )
        
        cat_pipeline = Pipeline(
                steps=[
                    ('imputer', SimpleImputer(strategy='most_frequent')),
                    ('ordinalencoder', OrdinalEncoder(categories=[
                        gender_categories,
                        martial_categories,
                        education_categories,
                        occupation_categories,
                        location_categories,
                        policy_categories,
                        feedback_categories,
                        smoking_categories,
                        exercise_categories,
                        property_categories
                    ])),
                    ('scaler', StandardScaler())
                ]
            )
        
        date_pipeline=Pipeline([
            ('extract_days_since_today',FunctionTransformer(self.extract_days_since_today, validate=False, feature_names_out=self.get_date_feature_names))
        ])
        
        preprocessor = ColumnTransformer([
                ('num_pipeline', num_pipeline, numerical_cols),
                ('cat_pipeline', cat_pipeline, categorical_cols),
                ('date_pipeline', date_pipeline, date_col)
            ])
        
        return preprocessor
    
    def extract_days_since_today(self, df):

        df = df.copy()
        # Convert to datetime with specified format
        df['Policy Start Date'] = pd.to_datetime(df['Policy Start Date'], format='%d-%m-%Y', errors='coerce')
        
        # Fill any NaN dates with a default date (e.g., today)
        df['Policy Start Date'] = df['Policy Start Date'].fillna(pd.Timestamp.today())
            
        today_date = pd.Timestamp.today()
        df['days_since_today'] = (today_date-df['Policy Start Date']).dt.days
        return df[['days_since_today']]
    
    def get_date_feature_names(self, input_features, output_features=None):
        """Returns feature names for the date transformer"""
        return ['days_since_today']
    
    def initialize_data_transformation(self, train_path, test_path):
    
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)

        preprocessing_obj = self.initiate_data_transformation()

        target_column_name = 'Premium Amount'
        drop_columns = [target_column_name, 'id']
        
        input_feature_train_df = train_df.drop(columns=drop_columns, axis=1)
        target_feature_train_df = train_df[target_column_name]

        input_feature_test_df = test_df.drop(columns=drop_columns, axis=1)
        target_feature_test_df = test_df[target_column_name]

        input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
        input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(self.preprocessor_obj_file_path), exist_ok=True)
        
        # Save preprocessor to file
        with open(self.preprocessor_obj_file_path, 'wb') as f:
            pickle.dump(preprocessing_obj, f)
            
        print(f"Preprocessor saved to {self.preprocessor_obj_file_path}")

        train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
        test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

        return train_arr,test_arr