
# Smart Premium ML: Insurance Premium Prediction App

## Overview
This project is an Insurance Premium Prediction application built using Streamlit, MLflow, and scikit-learn. The app allows users to predict insurance premiums based on various customer attributes including age, income, health factors, and policy details.

The application includes features for model training, prediction, and MLflow integration for experiment tracking and model management.

## Features
- **Interactive Premium Prediction** - Enter customer details to receive instant premium estimates
- **Model Training** - Train new models directly from the UI with the latest data
- **MLflow Integration** - Track experiments, compare models, and select the best model for prediction
- **Multi-tab Interface** - Separate tabs for prediction, training, and MLflow management
- **Flexible Model Selection** - Use either locally trained models or models from MLflow experiments

## Prerequisites
### Install Required Packages
Before running the application, you must have Python installed on your system along with the necessary packages.

```
pip install -r requirements.txt
```

## Installation & Setup
### Start the Application
Run the Streamlit app:
```
streamlit run streamlit_app.py
```

## How the Program Works
### Application Initialization
- When launched, the app connects to an optional MLflow server for experiment tracking
- The app can start a local MLflow server if one isn't already running
- Models are stored in the artifacts directory or tracked via MLflow

### Premium Prediction (Predict Premium Tab)
- Users input customer details including age, income, health metrics, and policy information
- The app processes inputs and applies feature transformations
- Users can select between the local trained model or an MLflow model
- The system displays the predicted premium amount

### Model Training (Train Model Tab)
- Users can train a new model using the latest data with a single click
- The app runs the complete pipeline: data ingestion, transformation, model training, and evaluation
- Multiple models are trained and compared (Linear Regression, Decision Tree, Random Forest, XGBoost)
- The best performing model is saved for future predictions

### MLflow Integration (MLflow Settings Tab)
- Start/stop MLflow server for experiment tracking
- View available model runs and their performance metrics
- Select specific models by Run ID for prediction
- Access the MLflow dashboard for detailed experiment analysis

## Usage Guide
1. **Predict Insurance Premium**
   - Enter customer details in the Predict Premium tab
   - Optionally select a model from MLflow
   - Click "Predict Insurance Premium"

2. **Train New Models**
   - Navigate to the Train Model tab
   - Click "Train Model" to start the training process
   - Wait for the process to complete and view results

3. **Manage MLflow**
   - Use the MLflow Settings tab to start/stop the MLflow server
   - List available models and copy Run IDs for prediction
   - Access the MLflow dashboard for deeper analysis

## Technologies Used
- Python
- Streamlit (Frontend & UI)
- MLflow (Experiment Tracking)
- scikit-learn & XGBoost (Machine Learning)
- Pandas & NumPy (Data Processing)

## Project Structure
- `streamlit_app.py` - Main Streamlit application
- `pipeline/` - Contains prediction and training pipelines
- `components/` - Modular components for data processing and model training
- `artifacts/` - Stores trained models and preprocessors
- `data/` - Sample training data
- `mlruns/` - MLflow experiment tracking data

## Author
Developed by Varun

Email: darklususnaturae@gmail.com