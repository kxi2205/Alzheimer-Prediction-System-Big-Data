#!/usr/bin/env python3
"""
Save the trained XGBoost model for API use
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import joblib
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the preprocessed data"""
    try:
        # Load training data
        X_train = pd.read_csv('outputs/preprocessed/X_train.csv')
        y_train = pd.read_csv('outputs/preprocessed/y_train.csv').values.ravel()
        
        return X_train, y_train
    except Exception as e:
        print(f"Error loading data: {e}")
        return None, None

def train_and_save_model():
    """Train XGBoost model and save it"""
    
    # Load data
    X_train, y_train = load_data()
    if X_train is None:
        print("Failed to load training data")
        return
    
    print(f"Training data shape: {X_train.shape}")
    print(f"Features: {list(X_train.columns)}")
    
    # Define hyperparameters (optimal from our grid search)
    best_params = {
        'colsample_bytree': 1.0,
        'gamma': 0,
        'learning_rate': 0.1,
        'max_depth': 5,
        'n_estimators': 200,
        'subsample': 0.8,
        'random_state': 42
    }
    
    # Train the model
    print("Training XGBoost model...")
    model = XGBClassifier(**best_params)
    model.fit(X_train, y_train)
    
    # Save the model
    models_dir = Path('outputs/models')
    models_dir.mkdir(exist_ok=True)
    
    model_path = models_dir / 'xgboost_model.pkl'
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Save feature names
    config_dir = Path('outputs/config')
    config_dir.mkdir(exist_ok=True)
    
    feature_names_path = config_dir / 'final_features.json'
    with open(feature_names_path, 'w') as f:
        json.dump(list(X_train.columns), f, indent=2)
    print(f"Feature names saved to: {feature_names_path}")
    
    # Test the model
    print("\nModel training completed successfully!")
    print(f"Model type: {type(model)}")
    print(f"Number of features: {len(X_train.columns)}")
    
    # Make a sample prediction to test
    sample_prediction = model.predict_proba(X_train.iloc[:1])
    print(f"Sample prediction probability: {sample_prediction[0]}")

if __name__ == "__main__":
    train_and_save_model()