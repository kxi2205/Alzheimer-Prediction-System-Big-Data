"""
Verify Streamlit Setup
======================
This script checks if all required files and dependencies are in place.
"""

import sys
from pathlib import Path
import importlib

def check_dependencies():
    """Check if required Python packages are installed"""
    required_packages = [
        'streamlit',
        'pandas',
        'numpy',
        'sklearn',
        'xgboost',
        'joblib',
        'plotly'
    ]
    
    print("Checking dependencies...")
    missing = []
    
    for package in required_packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} - NOT INSTALLED")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements_streamlit.txt")
        return False
    else:
        print("\n✓ All dependencies installed!")
        return True

def check_model_files():
    """Check if required model files exist"""
    print("\nChecking model files...")
    
    project_root = Path(__file__).parent
    
    files_to_check = [
        ('outputs/models/xgboost_model.pkl', 'XGBoost Model', True),
        ('outputs/config/final_features.json', 'Feature Configuration', False),
        ('streamlit_app.py', 'Streamlit App', True),
        ('.streamlit/config.toml', 'Streamlit Config', False)
    ]
    
    all_critical_exist = True
    
    for file_path, description, critical in files_to_check:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"✓ {description}: {file_path}")
        else:
            symbol = "✗" if critical else "⚠"
            print(f"{symbol} {description}: {file_path} - NOT FOUND")
            if critical:
                all_critical_exist = False
    
    if not all_critical_exist:
        print("\n⚠️ Critical files missing!")
        print("Run the model training notebooks to generate the model file.")
        return False
    else:
        print("\n✓ All critical files present!")
        return True

def check_model_loading():
    """Try to load the model"""
    print("\nTesting model loading...")
    
    try:
        import joblib
        project_root = Path(__file__).parent
        model_path = project_root / 'outputs' / 'models' / 'xgboost_model.pkl'
        
        if not model_path.exists():
            print("✗ Model file not found, skipping load test")
            return False
        
        model = joblib.load(model_path)
        print(f"✓ Model loaded successfully!")
        print(f"  Model type: {type(model).__name__}")
        
        if hasattr(model, 'n_features_in_'):
            print(f"  Expected features: {model.n_features_in_}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error loading model: {str(e)}")
        return False

def main():
    print("=" * 60)
    print("Streamlit Setup Verification")
    print("=" * 60)
    
    deps_ok = check_dependencies()
    files_ok = check_model_files()
    model_ok = check_model_loading()
    
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    if deps_ok and files_ok and model_ok:
        print("✓ All checks passed!")
        print("\nYou can now run the Streamlit app with:")
        print("  streamlit run streamlit_app.py")
    else:
        print("⚠️ Some checks failed. Please resolve the issues above.")
        if not deps_ok:
            print("\n1. Install dependencies:")
            print("   pip install -r requirements_streamlit.txt")
        if not files_ok:
            print("\n2. Generate model files:")
            print("   Run the Jupyter notebooks in the notebooks/ folder")
    
    print("=" * 60)

if __name__ == "__main__":
    main()
