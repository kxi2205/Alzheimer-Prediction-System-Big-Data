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
        
        # Create final dataframe with model features - use actual input values!
        final_df = pd.DataFrame(index=[0])
        
        # Map input data to model features directly
        for feature in model_features:
            if feature == 'CognitiveDeclineScore':
                final_df[feature] = df['CognitiveDeclineScore'].iloc[0]
            elif feature == 'TotalSymptomCount':
                final_df[feature] = df['TotalSymptomCount'].iloc[0]
            elif feature == 'MemoryComplaints':
                final_df[feature] = float(1 if data.get('MemoryComplaints', False) else 0)
            elif feature == 'BehavioralProblems':
                final_df[feature] = float(1 if data.get('BehavioralProblems', False) else 0)
            else:
                # Get the actual input value - don't default to 0!
                if feature in data and data[feature] is not None:
                    try:
                        final_df[feature] = float(data[feature])
                    except (ValueError, TypeError):
                        # Only use defaults if conversion fails
                        defaults = {
                            'Age': 65, 'BMI': 25, 'SleepQuality': 7,
                            'SystolicBP': 120, 'DiastolicBP': 80,
                            'CholesterolTotal': 200, 'CholesterolHDL': 50,
                            'CholesterolTriglycerides': 150, 'FunctionalAssessment': 8, 'ADL': 9
                        }
                        final_df[feature] = float(defaults.get(feature, 0))
                else:
                    # Only use defaults if the feature is truly missing from input
                    defaults = {
                        'Age': 65, 'BMI': 25, 'SleepQuality': 7,
                        'SystolicBP': 120, 'DiastolicBP': 80,
                        'CholesterolTotal': 200, 'CholesterolHDL': 50,
                        'CholesterolTriglycerides': 150, 'FunctionalAssessment': 8, 'ADL': 9
                    }
                    final_df[feature] = float(defaults.get(feature, 0))
        
        return final_df
        
    except Exception as e:
        raise Exception(f"Error preprocessing input: {str(e)}")

def calculate_risk_score_and_category(probability, patient_data):
    """Convert probability to 0-10 risk score with clinical adjustments"""
    # Base score: probability * 10
    base_score = probability * 10
    
    # Clinical adjustments
    adjustment_score = 0
    adjustments = []
    
    # Age > 75: +0.5 points
    age = float(patient_data.get('Age', 0))
    if age > 75:
        adjustment_score += 0.5
        adjustments.append("Advanced age (>75 years): +0.5")
    
    # MMSE < 24: +1.0 points
    mmse = float(patient_data.get('MMSE', 30))
    if mmse < 24:
        adjustment_score += 1.0
        adjustments.append("Cognitive impairment (MMSE<24): +1.0")
    
    # Family history: +0.5 points
    if patient_data.get('FamilyHistoryAlzheimers', 0):
        adjustment_score += 0.5
        adjustments.append("Family history of Alzheimer's: +0.5")
    
    # Multiple cardiovascular risks: +0.5 points
    cv_risks = [
        patient_data.get('Hypertension', 0),
        patient_data.get('Diabetes', 0), 
        patient_data.get('CardiovascularDisease', 0)
    ]
    cv_risk_count = sum(cv_risks)
    if cv_risk_count >= 2:
        adjustment_score += 0.5
        adjustments.append(f"Multiple cardiovascular risks ({cv_risk_count}): +0.5")
    
    # High symptom count (>3): +0.5 points
    symptoms = [
        patient_data.get('MemoryComplaints', 0),
        patient_data.get('BehavioralProblems', 0),
        patient_data.get('Confusion', 0),
        patient_data.get('Disorientation', 0),
        patient_data.get('PersonalityChanges', 0),
        patient_data.get('DifficultyCompletingTasks', 0),
        patient_data.get('Forgetfulness', 0)
    ]
    symptom_count = sum(symptoms)
    if symptom_count > 3:
        adjustment_score += 0.5
        adjustments.append(f"High symptom count ({symptom_count}): +0.5")
    
    # Final score (capped at 10)
    final_score = min(base_score + adjustment_score, 10.0)
    
    # Risk categories based on 0-10 scale
    if final_score <= 3:
        risk_category = "Low Risk"
    elif final_score <= 6:
        risk_category = "Moderate Risk" 
    else:
        risk_category = "High Risk"
    
    return final_score, risk_category, adjustments

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
        
        # Calculate risk score and category using 0-10 scale
        final_risk_score, risk_category, adjustments = calculate_risk_score_and_category(risk_score, input_data)
        
        # Prepare response
        result = {
            "score": round(final_risk_score, 1),
            "risk": risk_category,
            "model_used": "XGBoost",
            "base_probability": round(risk_score, 3),
            "adjustments": adjustments,
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