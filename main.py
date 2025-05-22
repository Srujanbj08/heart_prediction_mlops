import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from data_processing import preprocess_data, create_pipeline, save_pipeline, encode_labels
from ml_functions import train_model, evaluate
from helper_functions import log_info, log_error

def main():
    """Main function to train the heart disease prediction model"""
    try:
        # Create directories if they don't exist
        os.makedirs("artifacts", exist_ok=True)
        os.makedirs("data/output", exist_ok=True)
        
        # Load data - using the Cleveland heart disease dataset
        log_info("Loading dataset...")
        try:
            # Try loading from local file first
            file_path = "C:/Users/DHARSHAN KUMAR B J/Music/heart-disease-prediction/data/raw/prediction.csv"
            if os.path.exists(file_path):
                df = pd.read_csv(file_path)
            else:
                # If not found, load from UCI repository
                df = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data", 
                                 header=None, 
                                 names=['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 
                                        'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target'])
                
                # Save to local file for future use
                os.makedirs("data/raw", exist_ok=True)
                df.to_csv(file_path, index=False)
                log_info(f"Dataset saved to {file_path}")
        except Exception as e:
            log_error(f"Error loading dataset: {str(e)}")
            return
        
        # Clean data
        log_info("Preprocessing data...")
        df = df.replace('?', np.nan)
        # Convert columns to appropriate types
        for col in ['ca', 'thal']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Drop rows with missing values for simplicity
        df = df.dropna()
        
        # For the target, we'll consider 0 as no disease and 1-4 as disease
        if 'target' in df.columns:
            df['target'] = df['target'].apply(lambda x: 'Disease' if x > 0 else 'No Disease')
        
        # Preprocess the data
        df_processed = preprocess_data(df)
        
        # Split features and target
        X = df_processed.drop("target", axis=1)
        y = df_processed["target"]
        
        # Encode labels
        log_info("Encoding labels...")
        y_encoded = encode_labels(y)
        
        # Create and fit preprocessing pipeline
        log_info("Creating preprocessing pipeline...")
        pipeline = create_pipeline(X)
        X_transformed = pipeline.fit_transform(X)
        save_pipeline(pipeline)
        
        # Split data
        log_info("Splitting data into train and validation sets...")
        X_train, X_val, y_train, y_val = train_test_split(
            X_transformed, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )
        
        # Train model
        log_info("Training model...")
        train_model(X_train, y_train)
        
        # Evaluate model
        log_info("Evaluating model...")
        cm, acc, report = evaluate(X_val, y_val)
        
        if acc is not None:
            log_info(f"Model training complete with validation accuracy: {acc:.4f}")
            log_info(f"Classification Report:\n{report}")
        else:
            log_error("Model evaluation failed.")
            
    except Exception as e:
        log_error(f"An error occurred during training: {str(e)}")

if __name__ == "__main__":
    log_info("Starting model training process...")
    main()
    log_info("Model training process completed.")