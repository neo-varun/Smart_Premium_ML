import streamlit as st
import pandas as pd
from pipeline.prediction_pipeline import PredictPipeline, CustomData
import sys
import os
import socket
import subprocess
import time

# Function to check if MLflow server is running
def is_port_in_use(port):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        return s.connect_ex(('localhost', port)) == 0

# Function to start MLflow server
def start_mlflow_server():
    mlflow_cmd = "mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000"
    if not is_port_in_use(5000):
        subprocess.Popen(mlflow_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        time.sleep(2)
        return True
    return False

# Streamlit page title
st.set_page_config(page_title="Insurance Premium Predictor", layout="centered")

# Streamlit App Header
st.title("🚗 Insurance Premium Prediction App")

# Add tabs for prediction, training, and MLflow
tab1, tab2, tab3 = st.tabs(["Predict Premium", "Train Model", "MLflow Settings"])

with tab1:
    st.write("Enter the details below to predict the insurance premium.")
    
    # Add MLflow model option
    use_mlflow_model = st.checkbox("Use model from MLflow", value=False)
    
    selected_run_id = None
    if use_mlflow_model:
        if is_port_in_use(5000):
            # Simple text input for run ID
            selected_run_id = st.text_input(
                "Enter MLflow Run ID",
                help="Enter the run ID of the model you want to use for prediction (view in MLflow dashboard)"
            )
            
            if selected_run_id:
                st.info(f"Will use model from run ID: {selected_run_id}")
            else:
                st.warning("Please enter a valid run ID or uncheck the 'Use model from MLflow' option")
        else:
            st.error("MLflow server is not running. Start it from the MLflow Settings tab.")
            use_mlflow_model = False
    
    st.subheader("Customer Information")
    
    # User input fields
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    annual_income = st.number_input("Annual Income ($)", min_value=10000, max_value=500000, value=50000)
    num_dependents = st.number_input("Number of Dependents", min_value=0, max_value=10, value=1)
    health_score = st.number_input("Health Score (0-100)", min_value=0, max_value=100, value=75)
    previous_claims = st.number_input("Previous Claims", min_value=0, max_value=50, value=2)
    vehicle_age = st.number_input("Vehicle Age (Years)", min_value=0, max_value=30, value=5)
    credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=700)
    insurance_duration = st.number_input("Insurance Duration (Years)", min_value=1, max_value=30, value=10)
    policy_start_date = st.date_input("Policy Start Date")

    # Dropdown fields for categorical data
    gender = st.selectbox("Gender", ["Male", "Female"])
    marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
    education_level = st.selectbox("Education Level", ["High School", "Bachelor's", "Master's", "PhD"])
    occupation = st.selectbox("Occupation", ["Employed", "Self-Employed", "Unemployed"])
    location = st.selectbox("Location", ["Urban", "Suburban", "Rural"])
    policy_type = st.selectbox("Policy Type", ["Basic", "Comprehensive", "Premium"])
    customer_feedback = st.selectbox("Customer Feedback", ["Poor", "Average", "Good"])
    smoking_status = st.selectbox("Smoking Status", ["No", "Yes"])
    exercise_frequency = st.selectbox("Exercise Frequency", ["Daily", "Weekly", "Monthly", "Rarely"])
    property_type = st.selectbox("Property Type", ["House", "Apartment", "Condo"])

    # Predict Button
    if st.button("Predict Insurance Premium"):
        # Convert user input to DataFrame format
        # Format date properly to DD-MM-YYYY format
        formatted_date = policy_start_date.strftime('%d-%m-%Y')
        
        user_data = CustomData(
            age, annual_income, num_dependents, health_score, previous_claims, vehicle_age,
            credit_score, insurance_duration, formatted_date, gender, marital_status,
            education_level, occupation, location, policy_type, customer_feedback,
            smoking_status, exercise_frequency, property_type
        )
        
        input_df = user_data.get_data_as_dataframe()
        
        # Prediction Pipeline
        pipeline = PredictPipeline()
        
        # Use MLflow model if selected
        if use_mlflow_model and selected_run_id:
            try:
                with st.spinner(f"Loading model from MLflow Run ID: {selected_run_id}..."):
                    # Add explicit log for verification
                    st.info(f"Attempting to load model from MLflow run: {selected_run_id}")
                    prediction = pipeline.predict(input_df, run_id=selected_run_id)
                    st.success(f"💰 Predicted Insurance Premium: **${round(prediction[0], 2)}**")
                    st.success(f"✅ Successfully used model from MLflow Run ID: **{selected_run_id}**")
            except Exception as e:
                st.error(f"⚠️ Error loading model from MLflow: {e}")
                st.info("Check that MLflow server is running in the MLflow Settings tab.")
        else:
            try:
                with st.spinner("Making prediction with local model..."):
                    prediction = pipeline.predict(input_df)
                    st.success(f"💰 Predicted Insurance Premium: **${round(prediction[0], 2)}**")
                    st.info("Used local trained model from artifacts directory")
            except Exception as e:
                st.error(f"⚠️ Error during prediction: {e}")

with tab2:
    st.write("### Model Training")
    st.write("Click the button below to train the model with the latest data.")
    
    if st.button("Train Model"):
        with st.spinner("Training model in progress... This may take a few minutes."):
            try:
                # Execute the training pipeline script
                from pipeline.training_pipeline import train_model
                result = train_model()
                st.success("✅ Model training completed successfully!")
                st.info("The new model is now ready for predictions.")
            except Exception as e:
                st.error(f"❌ An error occurred during model training: {e}")

with tab3:
    st.write("### MLflow Server Settings")
    
    # Check if MLflow server is running
    if is_port_in_use(5000):
        st.success("✅ MLflow server is running on port 5000")
        
        # Add a section to list available runs
        st.subheader("Available MLflow Runs")
        if st.button("List Available Runs"):
            try:
                # Import MLflow here to avoid errors if it's not installed
                import mlflow
                from mlflow.tracking import MlflowClient
                
                # Set the tracking URI and get client
                mlflow.set_tracking_uri('http://localhost:5000')
                client = MlflowClient()
                
                # Get all experiments
                experiments = client.search_experiments()
                
                if not experiments:
                    st.info("No experiments found in MLflow. Train a model first.")
                else:
                    # Display runs from each experiment
                    for experiment in experiments:
                        st.write(f"**Experiment:** {experiment.name} (ID: {experiment.experiment_id})")
                        
                        # Get runs for this experiment
                        runs = client.search_runs(experiment_ids=[experiment.experiment_id])
                        
                        if not runs:
                            st.write("No runs found for this experiment.")
                        else:
                            # Create a table of runs
                            run_data = []
                            for run in runs:
                                run_data.append({
                                    "Run ID": run.info.run_id,
                                    "Status": run.info.status,
                                    "Start Time": run.info.start_time,
                                    "RMSE": run.data.metrics.get("rmse", "N/A"),
                                    "R²": run.data.metrics.get("r2", "N/A")
                                })
                            
                            if run_data:
                                st.dataframe(run_data)
                                st.info("Use any of these Run IDs in the Predict Premium tab to load a model")
            except Exception as e:
                st.error(f"Error accessing MLflow: {e}")
                st.info("Make sure MLflow server is running properly and the mlflow package is installed")
    else:
        st.error("❌ MLflow server is not running")
        if st.button("Start MLflow Server"):
            if start_mlflow_server():
                st.success("✅ MLflow server started successfully!")
                st.rerun()
            else:
                st.error("Failed to start MLflow server. Try running it manually.")
    
    # Only show dashboard button if server is running
    if is_port_in_use(5000):
        st.write("Access the MLflow dashboard to monitor models:")
        st.markdown("""
        <div style="text-align: center; margin-top: 20px;">
            <a href="http://localhost:5000" target="_blank" style="
                background-color: #4CAF50;
                border: none;
                color: white;
                padding: 12px 24px;
                text-align: center;
                text-decoration: none;
                display: inline-block;
                font-size: 16px;
                margin: 4px 2px;
                cursor: pointer;
                border-radius: 8px;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);">
                Open MLflow Dashboard ↗
            </a>
        </div>
        """, unsafe_allow_html=True)
    
    # Show the MLflow command for reference
    with st.expander("MLflow Server Command"):
        st.code("mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000", language="bash")
        st.info("This command is used to start the MLflow tracking server with a SQLite database backend.")