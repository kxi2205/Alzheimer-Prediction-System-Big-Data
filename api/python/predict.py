import sys
import json
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

def load_model_and_preprocessors():
    """Load the trained XGBoost model and preprocessing components"""
    try:
        # Paths to model files
        models_dir = project_root / 'outputs' / 'models'
        
        # Try to load XGBoost model (should be saved from notebook)
        model_path = models_dir / 'xgboost_model.pkl'
        if not model_path.exists():
            # Fallback to any available model
            model_files = list(models_dir.glob('*.pkl'))
            if model_files:
                model_path = model_files[0]
            else:
                raise FileNotFoundError("No trained model found")
        
        model = joblib.load(model_path)
        
        # Load feature names from preprocessing
        config_dir = project_root / 'outputs' / 'config'
        feature_names_path = config_dir / 'final_features.json'
        
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                feature_names = json.load(f)
        else:
            # Default feature set based on our analysis
            feature_names = [
                'Age', 'Gender', 'BMI', 'SystolicBP', 'DiastolicBP',
                'CholesterolTotal', 'CholesterolLDL', 'CholesterolHDL', 'CholesterolTriglycerides',
                'Smoking', 'AlcoholConsumption', 'PhysicalActivity', 'DietQuality', 'SleepQuality',
                'FamilyHistoryAlzheimers', 'CardiovascularDisease', 'Diabetes', 'Depression',
                'HeadInjury', 'Hypertension', 'MMSE', 'FunctionalAssessment', 'ADL',
                'MemoryComplaints', 'BehavioralProblems', 'Confusion', 'Disorientation',
                'PersonalityChanges', 'DifficultyCompletingTasks', 'Forgetfulness'
            ]
        
        return model, feature_names
        
    except Exception as e:
        raise Exception(f"Error loading model: {str(e)}")

def preprocess_input(data, feature_names):
    """Preprocess input data to match model training format"""
    try:
        # Create DataFrame with the input data
        df = pd.DataFrame([data])
        
        # Feature Engineering (matching the preprocessing pipeline)
        
        # 1. Calculate CognitiveDeclineScore
        cognitive_symptoms = [
            'MemoryComplaints', 'BehavioralProblems', 'Confusion', 
            'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks', 
            'Forgetfulness'
        ]
        
        cognitive_score = 0
        for symptom in cognitive_symptoms:
            if symptom in data and data[symptom]:
                cognitive_score += 1
        
        # Add MMSE weight to cognitive decline score
        mmse_score = float(data.get('MMSE', 28))  # Ensure numeric conversion
        if mmse_score < 24:
            cognitive_score += 2  # Severe cognitive impairment
        elif mmse_score < 28:
            cognitive_score += 1  # Mild cognitive impairment
            
        df['CognitiveDeclineScore'] = cognitive_score
        
        # 2. Calculate TotalSymptomCount
        all_symptoms = cognitive_symptoms + [
            'CardiovascularDisease', 'Diabetes', 'Depression', 
            'HeadInjury', 'Hypertension', 'Smoking'
        ]
        
        total_symptoms = sum(1 for symptom in all_symptoms if data.get(symptom, False))
        df['TotalSymptomCount'] = total_symptoms
        
        # 3. Select only the features that were used in the final model
        model_features = [
            'Age', 'BMI', 'SleepQuality', 'SystolicBP', 'DiastolicBP',
            'CholesterolTotal', 'CholesterolHDL', 'CholesterolTriglycerides',
            'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems',
            'ADL', 'CognitiveDeclineScore', 'TotalSymptomCount'
        ]
        
        # Create final dataframe with model features
        final_df = pd.DataFrame()
        for feature in model_features:
            if feature in ['CognitiveDeclineScore', 'TotalSymptomCount']:
                final_df[feature] = df[feature]
            elif feature == 'MemoryComplaints':
                final_df[feature] = 1 if data.get('MemoryComplaints', False) else 0
            elif feature == 'BehavioralProblems':
                final_df[feature] = 1 if data.get('BehavioralProblems', False) else 0
            else:
                # Ensure numeric conversion for all other features
                value = data.get(feature, 0)
                try:
                    final_df[feature] = float(value) if value is not None else 0
                except (ValueError, TypeError):
                    final_df[feature] = 0
        
        # Handle any missing values with reasonable defaults
        defaults = {
            'Age': 65,
            'BMI': 25,
            'SleepQuality': 7,
            'SystolicBP': 120,
            'DiastolicBP': 80,
            'CholesterolTotal': 200,
            'CholesterolHDL': 50,
            'CholesterolTriglycerides': 150,
            'FunctionalAssessment': 8,
            'ADL': 9
        }
        
        for feature in model_features:
            try:
                current_value = final_df[feature].iloc[0]
                # Check if value is missing, NaN, or effectively zero
                if pd.isna(current_value) or current_value == 0:
                    if feature in defaults:
                        final_df[feature] = float(defaults[feature])
                    elif feature in ['MemoryComplaints', 'BehavioralProblems']:
                        final_df[feature] = 0.0
                else:
                    # Ensure the value is properly converted to float
                    final_df[feature] = float(current_value)
            except (ValueError, TypeError, IndexError):
                # Fallback to defaults if conversion fails
                if feature in defaults:
                    final_df[feature] = float(defaults[feature])
                else:
                    final_df[feature] = 0.0
        
        return final_df
        
    except Exception as e:
        raise Exception(f"Error preprocessing input: {str(e)}")

def calculate_risk_category(probability):
    """Convert probability to risk category"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Moderate"
    else:
        return "High"

def main():
    try:
        # Read input from stdin
        input_data = json.loads(sys.stdin.read())
        
        # Load model and preprocessors
        model, feature_names = load_model_and_preprocessors()
        
        # Preprocess input data
        processed_data = preprocess_input(input_data, feature_names)
        
        # Make prediction
        if hasattr(model, 'predict_proba'):
            # For classifiers with probability prediction
            probability = model.predict_proba(processed_data)[0]
            if len(probability) > 1:
                # Binary classification - take probability of positive class
                risk_score = float(probability[1])
            else:
                risk_score = float(probability[0])
        else:
            # For classifiers without probability prediction
            prediction = model.predict(processed_data)[0]
            risk_score = float(prediction)
        
        # Calculate risk category
        risk_category = calculate_risk_category(risk_score)
        
        # Prepare response
        result = {
            "score": risk_score,
            "risk": risk_category,
            "model_used": "XGBoost",
            "confidence": "High" if abs(risk_score - 0.5) > 0.3 else "Medium"
        }
        
        # Output result as JSON
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            "error": str(e),
            "score": 0.5,
            "risk": "Unknown",
            "model_used": "Fallback",
            "confidence": "Low"
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()