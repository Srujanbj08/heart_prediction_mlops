import os
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from helper_functions import log_info, log_error

# Define paths
ARTIFACTS_PATH = "C:/Users/DHARSHAN KUMAR B J/Music/heart-disease-prediction/artifacts"
os.makedirs(ARTIFACTS_PATH, exist_ok=True)
PIPELINE_PATH = os.path.join(ARTIFACTS_PATH, "heart_pipeline.pkl")
LABEL_ENCODER_PATH = os.path.join(ARTIFACTS_PATH, "heart_label_encoder.pkl")

def preprocess_data(df):
    """Clean and prepare the data"""
    # Make a copy to avoid modifying the original dataframe
    data = df.copy()
    
    # Map categorical values to more descriptive ones
    sex_mapping = {0: 'Female', 1: 'Male'}
    cp_mapping = {0: 'Typical', 1: 'Atypical', 2: 'Non-anginal', 3: 'Asymptomatic'}
    fbs_mapping = {0: 'No', 1: 'Yes'}
    restecg_mapping = {0: 'Normal', 1: 'ST-T Abnormality', 2: 'LV Hypertrophy'}
    exang_mapping = {0: 'No', 1: 'Yes'}
    slope_mapping = {0: 'Upsloping', 1: 'Flat', 2: 'Downsloping'}
    thal_mapping = {1: 'Normal', 2: 'Fixed Defect', 3: 'Reversible Defect'}
    
    # Apply mappings if columns are numeric
    if pd.api.types.is_numeric_dtype(data['sex']):
        data['sex'] = data['sex'].map(sex_mapping)
    if pd.api.types.is_numeric_dtype(data['cp']):
        data['cp'] = data['cp'].map(cp_mapping)
    if pd.api.types.is_numeric_dtype(data['fbs']):
        data['fbs'] = data['fbs'].map(fbs_mapping)
    if pd.api.types.is_numeric_dtype(data['restecg']):
        data['restecg'] = data['restecg'].map(restecg_mapping)
    if pd.api.types.is_numeric_dtype(data['exang']):
        data['exang'] = data['exang'].map(exang_mapping)
    if pd.api.types.is_numeric_dtype(data['slope']):
        data['slope'] = data['slope'].map(slope_mapping)
    if 'thal' in data.columns and pd.api.types.is_numeric_dtype(data['thal']):
        data['thal'] = data['thal'].map(thal_mapping)
    
    return data

def create_pipeline(X):
    """Create the preprocessing pipeline"""
    # Identify categorical and numerical columns
    categorical = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    log_info(f"Categorical features: {categorical}")
    log_info(f"Numerical features: {numeric}")
    
    # Create preprocessing steps
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical)
    ])
    
    # Create pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    log_info("Pipeline created successfully")
    return pipeline

def save_pipeline(pipeline):
    """Save preprocessing pipeline to disk"""
    try:
        with open(PIPELINE_PATH, 'wb') as f:
            pickle.dump(pipeline, f)
        log_info(f"Pipeline saved at {PIPELINE_PATH}")
    except Exception as e:
        log_error(f"Error saving pipeline: {str(e)}")

def encode_labels(y):
    """Encode target labels and save encoder"""
    try:
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)
        
        with open(LABEL_ENCODER_PATH, 'wb') as f:
            pickle.dump(le, f)
        log_info(f"Label encoder saved at {LABEL_ENCODER_PATH}")
        return y_encoded
    except Exception as e:
        log_error(f"Error encoding labels: {str(e)}")
        return None