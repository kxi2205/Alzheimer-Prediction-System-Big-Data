# Alzheimer's Disease Prediction System

A professional, full-stack medical assessment application designed to predict the risk of Alzheimer's disease using clinical data and advanced machine learning.

## 🚀 Architecture
- **Frontend**: Next.js 15+ (React 19, Tailwind CSS, TypeScript)
- **Backend API**: Next.js API Routes (Node.js)
- **ML Engine**: Python 3.12 (XGBoost, Scikit-learn, Pandas)
- **Deployment**: Integrated prediction pipeline using the `spawn` process for real-time inference.

## 📋 Features
- **Modern Assessment Form**: A responsive, 6-step clinical assessment form with validation.
- **Real-time Prediction**: Instant risk scoring (0-10 scale) powered by an optimized XGBoost model.
- **Clinical Adjustments**: Rule-based logic that adjusts AI predictions based on critical clinical markers (e.g., family history, memory complaints).
- **Layman-Friendly Results**: Clear risk categorization (Low/Moderate/High) with interpretation and recommendations.

## 🛠️ Setup & Installation

### 1. Prerequisites
- Node.js (v18+)
- Python (v3.10+)

### 2. Frontend Setup
```bash
npm install
npm run dev
```
The application will be available at `http://localhost:3000`.

### 3. Backend Setup (Python)
Install the required machine learning libraries:
```bash
pip install -r requirements.txt
```

## 🧪 Machine Learning Pipeline
The system uses an optimized XGBoost model as the primary engine, with Random Forest and SVM variants also validated during development.

### Model Performance (Test Set)
| Model | Accuracy | Precision | Recall | F1 Score | ROC AUC |
|-------|----------|-----------|--------|----------|---------|
| **Random Forest** | 94.1% | 91.3% | 92.1% | 91.7% | 0.9505 |
| **SVM (RBF)** | 83.9% | 75.8% | 79.8% | 77.8% | 0.9033 |

- **Preprocessing**: Includes missing-value imputation (median/mode), categorical encoding, and feature scaling.
- **Training Data**: Located in `data/alzheimer_dataset.csv`.
- **Primary Model**: Saved in `outputs/models/xgboost_model.pkl`.

## 📁 Repository Structure
- `app/`: Next.js application routes and API logic.
- `components/`: Reusable React components (Assessment form, Result displays).
- `api/python/`: Core Python prediction scripts and model integration.
- `outputs/`: Model artifacts, configurations, and evaluation results.
- `scripts/`: Production-grade Python scripts for preprocessing and training.
- `notebooks/`: Jupyter notebooks for research and development.

## ⚖️ Disclaimer
*This system is a clinical decision support tool and should not replace professional medical judgment. All assessments should be reviewed by qualified healthcare providers.*
