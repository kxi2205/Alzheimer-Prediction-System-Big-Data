"""
Alzheimer's Disease Risk Assessment - Streamlit Application
============================================================
A comprehensive risk assessment tool for Alzheimer's disease using machine learning.
"""

import streamlit as st
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
import warnings
import plotly.graph_objects as go
import plotly.express as px

warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Alzheimer's Risk Assessment",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f2937;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .risk-card {
        padding: 2rem;
        border-radius: 1rem;
        margin: 1rem 0;
        text-align: center;
    }
    .risk-low {
        background-color: #d1fae5;
        border: 2px solid #10b981;
    }
    .risk-moderate {
        background-color: #fef3c7;
        border: 2px solid #f59e0b;
    }
    .risk-high {
        background-color: #fee2e2;
        border: 2px solid #ef4444;
    }
    .metric-card {
        background-color: #f9fafb;
        padding: 1rem;
        border-radius: 0.5rem;
        border: 1px solid #e5e7eb;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and configuration
@st.cache_resource
def load_model_and_config():
    """Load the trained model and configuration files"""
    try:
        project_root = Path(__file__).parent
        models_dir = project_root / 'outputs' / 'models'
        
        # Load XGBoost model
        model_path = models_dir / 'xgboost_model.pkl'
        if not model_path.exists():
            model_files = list(models_dir.glob('*.pkl'))
            if model_files:
                model_path = model_files[0]
            else:
                return None, None
        
        model = joblib.load(model_path)
        
        # Load feature names
        config_dir = project_root / 'outputs' / 'config'
        feature_names_path = config_dir / 'final_features.json'
        
        if feature_names_path.exists():
            with open(feature_names_path, 'r') as f:
                feature_names = json.load(f)
        else:
            feature_names = [
                'Age', 'BMI', 'SleepQuality', 'SystolicBP', 'DiastolicBP',
                'CholesterolTotal', 'CholesterolHDL', 'CholesterolTriglycerides',
                'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems',
                'ADL', 'CognitiveDeclineScore', 'TotalSymptomCount'
            ]
        
        return model, feature_names
        
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None

def preprocess_input(data):
    """Preprocess input data for model prediction"""
    # Calculate Cognitive Decline Score
    cognitive_symptoms = [
        'MemoryComplaints', 'BehavioralProblems', 'Confusion',
        'Disorientation', 'PersonalityChanges', 'DifficultyCompletingTasks',
        'Forgetfulness'
    ]
    
    cognitive_score = sum(1 for symptom in cognitive_symptoms if data.get(symptom, False))
    
    # Add MMSE weight
    mmse_score = float(data.get('MMSE', 28))
    if mmse_score < 24:
        cognitive_score += 2
    elif mmse_score < 28:
        cognitive_score += 1
    
    data['CognitiveDeclineScore'] = cognitive_score
    
    # Calculate Total Symptom Count
    all_symptoms = cognitive_symptoms + [
        'CardiovascularDisease', 'Diabetes', 'Depression',
        'HeadInjury', 'Hypertension', 'Smoking'
    ]
    
    data['TotalSymptomCount'] = sum(1 for symptom in all_symptoms if data.get(symptom, False))
    
    # Model features
    model_features = [
        'Age', 'BMI', 'SleepQuality', 'SystolicBP', 'DiastolicBP',
        'CholesterolTotal', 'CholesterolHDL', 'CholesterolTriglycerides',
        'FunctionalAssessment', 'MemoryComplaints', 'BehavioralProblems',
        'ADL', 'CognitiveDeclineScore', 'TotalSymptomCount'
    ]
    
    # Create feature dataframe
    final_df = pd.DataFrame(index=[0])
    for feature in model_features:
        if feature in ['CognitiveDeclineScore', 'TotalSymptomCount']:
            final_df[feature] = data[feature]
        elif feature in ['MemoryComplaints', 'BehavioralProblems']:
            final_df[feature] = float(1 if data.get(feature, False) else 0)
        else:
            final_df[feature] = float(data.get(feature, 0))
    
    return final_df

def calculate_risk_score(probability, patient_data):
    """Calculate risk score with clinical adjustments"""
    base_score = probability * 10
    adjustment_score = 0
    adjustments = []
    
    # Age > 75
    if patient_data.get('Age', 0) > 75:
        adjustment_score += 0.5
        adjustments.append("Advanced age (>75 years): +0.5")
    
    # MMSE < 24
    if patient_data.get('MMSE', 30) < 24:
        adjustment_score += 1.0
        adjustments.append("Cognitive impairment (MMSE<24): +1.0")
    
    # Family history
    if patient_data.get('FamilyHistoryAlzheimers', False):
        adjustment_score += 0.5
        adjustments.append("Family history of Alzheimer's: +0.5")
    
    # Cardiovascular risks
    cv_risks = sum([
        patient_data.get('Hypertension', False),
        patient_data.get('Diabetes', False),
        patient_data.get('CardiovascularDisease', False)
    ])
    if cv_risks >= 2:
        adjustment_score += 0.5
        adjustments.append(f"Multiple cardiovascular risks ({cv_risks}): +0.5")
    
    # High symptom count
    symptoms = sum([
        patient_data.get('MemoryComplaints', False),
        patient_data.get('BehavioralProblems', False),
        patient_data.get('Confusion', False),
        patient_data.get('Disorientation', False),
        patient_data.get('PersonalityChanges', False),
        patient_data.get('DifficultyCompletingTasks', False),
        patient_data.get('Forgetfulness', False)
    ])
    if symptoms > 3:
        adjustment_score += 0.5
        adjustments.append(f"High symptom count ({symptoms}): +0.5")
    
    final_score = min(base_score + adjustment_score, 10.0)
    
    # Risk category
    if final_score <= 3:
        risk_category = "Low Risk"
        risk_class = "risk-low"
    elif final_score <= 6:
        risk_category = "Moderate Risk"
        risk_class = "risk-moderate"
    else:
        risk_category = "High Risk"
        risk_class = "risk-high"
    
    return final_score, risk_category, risk_class, adjustments, base_score

def create_risk_gauge(score):
    """Create a gauge chart for risk score"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Risk Score", 'font': {'size': 24}},
        gauge={
            'axis': {'range': [None, 10], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 3], 'color': '#d1fae5'},
                {'range': [3, 6], 'color': '#fef3c7'},
                {'range': [6, 10], 'color': '#fee2e2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': score
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

def get_recommendations(risk_score, adjustments, patient_data):
    """Generate personalized recommendations"""
    recommendations = []
    
    if risk_score >= 7:
        recommendations += [
            "🔴 Seek immediate medical evaluation for comprehensive cognitive assessment",
            "📋 Consider referral to neurology or memory specialist",
            "🧠 Undergo detailed neuropsychological testing"
        ]
    elif risk_score >= 4:
        recommendations += [
            "🟡 Schedule enhanced cognitive screening within 6 months",
            "📊 Monitor cognitive function more frequently",
            "🏥 Discuss with primary care physician"
        ]
    else:
        recommendations += [
            "🟢 Continue regular health check-ups",
            "🧠 Maintain current cognitive health practices"
        ]
    
    # General recommendations
    recommendations += [
        "🏃‍♀️ Engage in regular physical exercise (150+ minutes/week)",
        "🧘‍♀️ Practice cognitive stimulation activities (puzzles, reading, social interaction)",
        "🥗 Follow Mediterranean diet rich in omega-3 fatty acids",
        "😴 Maintain good sleep hygiene (7-8 hours/night)",
        "🚭 Avoid smoking and limit alcohol consumption"
    ]
    
    # Specific recommendations based on adjustments
    if any("cardiovascular" in adj.lower() for adj in adjustments):
        recommendations.append("❤️ Focus on cardiovascular health management")
        recommendations.append("💊 Ensure optimal blood pressure and cholesterol control")
    
    if any("mmse" in adj.lower() for adj in adjustments):
        recommendations.append("🧠 Consider cognitive training programs")
        recommendations.append("📚 Engage in mentally stimulating activities daily")
    
    return recommendations

def main():
    # Header
    st.markdown('<div class="main-header">🧠 Alzheimer\'s Disease Risk Assessment</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Complete this comprehensive assessment to evaluate your risk factors for Alzheimer\'s disease using advanced machine learning models.</div>', unsafe_allow_html=True)
    
    # Load model
    model, feature_names = load_model_and_config()
    
    if model is None:
        st.error("⚠️ Model not found. Please ensure the model files are in the outputs/models directory.")
        return
    
    # Sidebar for navigation
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/brain.png", width=100)
        st.title("Navigation")
        page = st.radio("Select Page", ["Assessment", "About", "Help"])
        
        st.markdown("---")
        st.info("This tool is for educational and informational purposes only. Always consult with healthcare professionals for medical advice.")
    
    if page == "Assessment":
        show_assessment_page(model, feature_names)
    elif page == "About":
        show_about_page()
    else:
        show_help_page()

def show_assessment_page(model, feature_names):
    """Display the main assessment form"""
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📋 Personal Info",
        "💪 Physical Health",
        "🏃 Lifestyle",
        "🏥 Medical History",
        "🧠 Cognitive Assessment"
    ])
    
    # Initialize session state for form data
    if 'form_data' not in st.session_state:
        st.session_state.form_data = {}
    
    # Tab 1: Personal Information
    with tab1:
        st.header("Personal Information")
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age", min_value=18, max_value=120, value=65, help="Your current age in years")
            gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        
        with col2:
            ethnicity = st.selectbox("Ethnicity", ["Caucasian", "African American", "Asian", "Other"])
            education = st.selectbox("Education Level", ["None", "High School", "Bachelor's", "Higher"])
        
        st.session_state.form_data.update({
            'Age': age,
            'Gender': gender,
            'Ethnicity': ethnicity,
            'EducationLevel': education
        })
    
    # Tab 2: Physical Health
    with tab2:
        st.header("Physical Health Metrics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Body Metrics")
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
            weight = st.number_input("Weight (kg)", min_value=30, max_value=300, value=70)
            bmi = weight / ((height/100) ** 2)
            st.metric("BMI", f"{bmi:.1f}")
            
            st.subheader("Blood Pressure")
            systolic = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=200, value=120)
            diastolic = st.number_input("Diastolic BP (mmHg)", min_value=40, max_value=130, value=80)
        
        with col2:
            st.subheader("Cholesterol Levels")
            chol_total = st.number_input("Total Cholesterol (mg/dL)", min_value=100, max_value=400, value=200)
            chol_ldl = st.number_input("LDL Cholesterol (mg/dL)", min_value=50, max_value=300, value=100)
            chol_hdl = st.number_input("HDL Cholesterol (mg/dL)", min_value=20, max_value=100, value=50)
            chol_trig = st.number_input("Triglycerides (mg/dL)", min_value=50, max_value=500, value=150)
        
        st.session_state.form_data.update({
            'BMI': bmi,
            'SystolicBP': systolic,
            'DiastolicBP': diastolic,
            'CholesterolTotal': chol_total,
            'CholesterolLDL': chol_ldl,
            'CholesterolHDL': chol_hdl,
            'CholesterolTriglycerides': chol_trig
        })
    
    # Tab 3: Lifestyle Factors
    with tab3:
        st.header("Lifestyle Factors")
        
        col1, col2 = st.columns(2)
        
        with col1:
            smoking = st.checkbox("Current Smoker")
            alcohol = st.slider("Alcohol Consumption (drinks/week)", 0, 20, 2)
            physical = st.slider("Physical Activity (hours/week)", 0, 20, 5)
        
        with col2:
            diet = st.slider("Diet Quality (1-10)", 1, 10, 5, help="1=Poor, 10=Excellent")
            sleep = st.slider("Sleep Quality (hours/night)", 0, 12, 7)
        
        st.session_state.form_data.update({
            'Smoking': smoking,
            'AlcoholConsumption': alcohol,
            'PhysicalActivity': physical,
            'DietQuality': diet,
            'SleepQuality': sleep
        })
    
    # Tab 4: Medical History
    with tab4:
        st.header("Medical History")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Family & Genetic")
            family_alz = st.checkbox("Family History of Alzheimer's")
            
            st.subheader("Chronic Conditions")
            cardiovascular = st.checkbox("Cardiovascular Disease")
            diabetes = st.checkbox("Diabetes")
            hypertension = st.checkbox("Hypertension")
        
        with col2:
            st.subheader("Other Conditions")
            depression = st.checkbox("Depression")
            head_injury = st.checkbox("History of Head Injury")
        
        st.session_state.form_data.update({
            'FamilyHistoryAlzheimers': family_alz,
            'CardiovascularDisease': cardiovascular,
            'Diabetes': diabetes,
            'Depression': depression,
            'HeadInjury': head_injury,
            'Hypertension': hypertension
        })
    
    # Tab 5: Cognitive Assessment
    with tab5:
        st.header("Cognitive Assessment")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Cognitive Scores")
            mmse = st.slider("MMSE Score", 0, 30, 28, help="Mini-Mental State Examination (0-30)")
            functional = st.slider("Functional Assessment", 0, 10, 8)
            adl = st.slider("Activities of Daily Living (ADL)", 0, 10, 9)
            
            st.info(f"""
            **MMSE Interpretation:**
            - 24-30: Normal
            - 18-23: Mild impairment
            - 0-17: Severe impairment
            """)
        
        with col2:
            st.subheader("Cognitive Symptoms")
            memory = st.checkbox("Memory Complaints")
            behavioral = st.checkbox("Behavioral Problems")
            confusion = st.checkbox("Confusion")
            disorientation = st.checkbox("Disorientation")
            personality = st.checkbox("Personality Changes")
            difficulty = st.checkbox("Difficulty Completing Tasks")
            forgetfulness = st.checkbox("Forgetfulness")
        
        st.session_state.form_data.update({
            'MMSE': mmse,
            'FunctionalAssessment': functional,
            'ADL': adl,
            'MemoryComplaints': memory,
            'BehavioralProblems': behavioral,
            'Confusion': confusion,
            'Disorientation': disorientation,
            'PersonalityChanges': personality,
            'DifficultyCompletingTasks': difficulty,
            'Forgetfulness': forgetfulness
        })
    
    # Submit button
    st.markdown("---")
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        if st.button("🔍 Calculate Risk Score", type="primary", use_container_width=True):
            with st.spinner("Analyzing your risk factors..."):
                # Preprocess data
                processed_data = preprocess_input(st.session_state.form_data)
                
                # Make prediction
                probability = model.predict_proba(processed_data)[0][1]
                
                # Calculate risk score
                final_score, risk_category, risk_class, adjustments, base_score = calculate_risk_score(
                    probability, st.session_state.form_data
                )
                
                # Display results
                st.markdown("---")
                st.markdown("## 📊 Risk Assessment Results")
                
                # Risk score card
                st.markdown(f"""
                <div class="risk-card {risk_class}">
                    <h2 style="margin: 0;">Risk Score: {final_score:.1f}/10</h2>
                    <h3 style="margin: 10px 0;">{risk_category}</h3>
                    <p style="margin: 0;">Base Probability: {probability:.1%}</p>
                </div>
                """, unsafe_allow_html=True)
                
                # Create columns for metrics and gauge
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.plotly_chart(create_risk_gauge(final_score), use_container_width=True)
                
                with col2:
                    st.markdown("### 📈 Score Breakdown")
                    st.metric("Base Model Score", f"{base_score:.1f}/10")
                    st.metric("Clinical Adjustments", f"+{final_score - base_score:.1f}")
                    st.metric("Final Risk Score", f"{final_score:.1f}/10")
                
                # Adjustments
                if adjustments:
                    st.markdown("### ⚕️ Clinical Adjustments Applied")
                    for adj in adjustments:
                        st.info(adj)
                
                # Recommendations
                st.markdown("### 💡 Personalized Recommendations")
                recommendations = get_recommendations(final_score, adjustments, st.session_state.form_data)
                
                for i, rec in enumerate(recommendations, 1):
                    st.markdown(f"{i}. {rec}")
                
                # Download report
                st.markdown("---")
                report = generate_report(final_score, risk_category, probability, adjustments, recommendations, st.session_state.form_data)
                st.download_button(
                    label="📥 Download Report",
                    data=report,
                    file_name="alzheimer_risk_assessment.txt",
                    mime="text/plain"
                )

def generate_report(score, category, probability, adjustments, recommendations, patient_data):
    """Generate a text report of the assessment"""
    report = f"""
ALZHEIMER'S DISEASE RISK ASSESSMENT REPORT
==========================================

Assessment Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

PATIENT INFORMATION
-------------------
Age: {patient_data.get('Age', 'N/A')}
Gender: {patient_data.get('Gender', 'N/A')}
Ethnicity: {patient_data.get('Ethnicity', 'N/A')}
Education: {patient_data.get('EducationLevel', 'N/A')}

RISK ASSESSMENT
---------------
Risk Score: {score:.1f}/10
Risk Category: {category}
Model Confidence: {probability:.1%}

CLINICAL ADJUSTMENTS
--------------------
"""
    if adjustments:
        for adj in adjustments:
            report += f"- {adj}\n"
    else:
        report += "- No clinical adjustments applied\n"
    
    report += "\nRECOMMENDATIONS\n"
    report += "---------------\n"
    for i, rec in enumerate(recommendations, 1):
        report += f"{i}. {rec}\n"
    
    report += """
DISCLAIMER
----------
This assessment is for educational and informational purposes only.
It should not be used as a substitute for professional medical advice,
diagnosis, or treatment. Always seek the advice of your physician or
other qualified health provider with any questions you may have regarding
a medical condition.
"""
    
    return report

def show_about_page():
    """Display information about the system"""
    st.header("About This System")
    
    st.markdown("""
    ### 🧠 Alzheimer's Disease Risk Assessment System
    
    This application uses advanced machine learning techniques to assess the risk
    of Alzheimer's disease based on multiple factors including:
    
    - **Demographics**: Age, gender, ethnicity, education
    - **Physical Health**: BMI, blood pressure, cholesterol levels
    - **Lifestyle**: Smoking, alcohol, exercise, diet, sleep
    - **Medical History**: Family history, chronic conditions
    - **Cognitive Function**: MMSE scores, cognitive symptoms
    
    ### 🤖 Machine Learning Model
    
    The system uses an **XGBoost classifier** trained on clinical data to predict
    Alzheimer's disease risk. The model achieves high accuracy and provides
    interpretable risk scores.
    
    ### 📊 Risk Score Calculation
    
    The risk score (0-10 scale) is calculated using:
    1. **Base Model Prediction**: Machine learning probability × 10
    2. **Clinical Adjustments**: Additional points for high-risk factors
    
    ### 🎯 Risk Categories
    
    - **Low Risk (0-3)**: Low probability of Alzheimer's disease
    - **Moderate Risk (3-6)**: Moderate probability, monitoring recommended
    - **High Risk (6-10)**: High probability, medical evaluation recommended
    
    ### ⚠️ Important Disclaimer
    
    This tool is designed for educational and research purposes only. It should
    not replace professional medical advice, diagnosis, or treatment. Always
    consult with qualified healthcare professionals for medical decisions.
    """)

def show_help_page():
    """Display help and instructions"""
    st.header("Help & Instructions")
    
    st.markdown("""
    ### 📝 How to Use This Tool
    
    1. **Navigate through the tabs** to complete each section of the assessment
    2. **Fill in all required information** accurately
    3. **Click "Calculate Risk Score"** to generate your risk assessment
    4. **Review your results** and personalized recommendations
    5. **Download the report** for your records
    
    ### 💡 Tips for Accurate Assessment
    
    - Provide accurate information for all fields
    - Consult medical records for precise measurements
    - Answer cognitive symptom questions honestly
    - Update the assessment periodically (every 6-12 months)
    
    ### 🏥 Understanding Your Results
    
    #### Risk Score
    Your risk score is calculated on a 0-10 scale, where higher scores indicate
    greater risk of Alzheimer's disease.
    
    #### Clinical Adjustments
    Additional points may be added based on specific high-risk factors such as:
    - Advanced age (>75 years)
    - Cognitive impairment (MMSE < 24)
    - Family history of Alzheimer's
    - Multiple cardiovascular risk factors
    - High symptom count
    
    #### Recommendations
    Personalized recommendations are provided based on your risk profile and
    include both preventive measures and medical follow-up suggestions.
    
    ### 📞 Need Help?
    
    If you have questions about:
    - **Technical issues**: Contact support
    - **Medical questions**: Consult your healthcare provider
    - **Understanding results**: Review the About page or speak with a medical professional
    
    ### 🔒 Privacy & Data Security
    
    All data entered is processed locally and is not stored or transmitted to
    external servers. Your privacy is protected.
    """)

if __name__ == "__main__":
    main()
