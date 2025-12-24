# Streamlit Cloud Deployment Troubleshooting

## ✅ Issue Fixed: ModuleNotFoundError for joblib

### Problem
```
ModuleNotFoundError: This app has encountered an error.
File "/mount/src/alzheimer-prediction-system-big-data/streamlit_app.py", line 11, in <module>
    import joblib
```

### Root Cause
Streamlit Cloud looks for `requirements.txt` in the root directory, but we initially created `requirements_streamlit.txt`.

### Solution
✅ Created `requirements.txt` in the root directory with all dependencies including `joblib>=1.3.0`

## 📋 Pre-Deployment Checklist

Before deploying to Streamlit Cloud, ensure:

### 1. ✅ Requirements File
- [ ] `requirements.txt` exists in root directory
- [ ] All packages are listed with versions
- [ ] File includes:
  - streamlit>=1.28.0
  - pandas>=2.0.0
  - numpy>=1.24.0
  - scikit-learn>=1.3.0
  - xgboost>=2.0.0
  - joblib>=1.3.0
  - plotly>=5.17.0

### 2. ✅ Model Files
- [ ] `outputs/models/xgboost_model.pkl` exists
- [ ] Model file is committed to repository
- [ ] Model size < 100MB (or use Git LFS)
- [ ] `.gitattributes` configured for binary files

### 3. ✅ Configuration Files
- [ ] `.streamlit/config.toml` exists
- [ ] Theme settings configured
- [ ] Server settings properly set

### 4. ✅ Main Application
- [ ] `streamlit_app.py` in root directory
- [ ] All imports use correct package names
- [ ] File paths are relative (use `Path(__file__).parent`)
- [ ] No hardcoded absolute paths

### 5. ✅ Git Repository
- [ ] All files committed and pushed
- [ ] Model files included (or in Git LFS)
- [ ] No sensitive data in repository
- [ ] `.gitignore` properly configured

## 🚀 Streamlit Cloud Deployment Steps

### Step 1: Push to GitHub

```bash
# Check status
git status

# Add all new files
git add requirements.txt
git add streamlit_app.py
git add .streamlit/config.toml
git add .gitattributes
git add outputs/models/xgboost_model.pkl
git add outputs/config/final_features.json

# Commit
git commit -m "Add Streamlit deployment files"

# Push
git push origin main
```

### Step 2: Deploy on Streamlit Cloud

1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click "New app"
4. Select repository: `yjhkdjsg/Alzheimer-Prediction-System-Big-Data`
5. Branch: `main`
6. Main file path: `streamlit_app.py`
7. Click "Deploy"

### Step 3: Monitor Deployment

Watch the deployment logs for:
- ✅ Requirements installation
- ✅ App startup
- ✅ Model loading
- ⚠️ Any errors or warnings

## 🐛 Common Issues & Solutions

### Issue 1: ModuleNotFoundError

**Error:**
```
ModuleNotFoundError: No module named 'joblib'
```

**Solution:**
- Ensure `requirements.txt` exists in root directory
- Verify package is listed: `joblib>=1.3.0`
- Check package name spelling
- Rebuild app on Streamlit Cloud

### Issue 2: Model File Not Found

**Error:**
```
FileNotFoundError: [Errno 2] No such file or directory: 'outputs/models/xgboost_model.pkl'
```

**Solution:**
- Verify model file is in repository
- Check file path is relative, not absolute
- Ensure model file was committed and pushed
- For large models (>100MB), use Git LFS:
  ```bash
  git lfs install
  git lfs track "*.pkl"
  git add .gitattributes
  git add outputs/models/xgboost_model.pkl
  git commit -m "Add model with LFS"
  git push
  ```

### Issue 3: Import Errors

**Error:**
```
ImportError: cannot import name 'xxx' from 'yyy'
```

**Solution:**
- Check package versions in `requirements.txt`
- Ensure compatible versions:
  - scikit-learn and xgboost compatibility
  - pandas and numpy compatibility
- Try pinning specific versions:
  ```
  xgboost==2.0.3
  scikit-learn==1.3.2
  ```

### Issue 4: Memory Limit Exceeded

**Error:**
```
MemoryError or app crashes
```

**Solution:**
- Optimize model size (compress, reduce features)
- Use model quantization
- Cache model loading: `@st.cache_resource`
- Contact Streamlit for higher limits (Community Cloud free tier: 1GB RAM)

### Issue 5: Slow App Performance

**Symptoms:**
- App takes long to load
- Predictions are slow

**Solution:**
- Use `@st.cache_resource` for model loading
- Use `@st.cache_data` for data processing
- Optimize preprocessing pipeline
- Reduce model complexity if needed
- Enable session state for form data

### Issue 6: CORS/Security Errors

**Error:**
```
Warning: CORS/XSRF configuration conflict
```

**Solution:**
Update `.streamlit/config.toml`:
```toml
[server]
enableCORS = true
enableXsrfProtection = true
```

### Issue 7: Path Issues (Windows vs Linux)

**Error:**
```
Path not found or permission denied
```

**Solution:**
Use `pathlib.Path` for cross-platform compatibility:
```python
from pathlib import Path
project_root = Path(__file__).parent
model_path = project_root / 'outputs' / 'models' / 'xgboost_model.pkl'
```

## 🔍 Debugging on Streamlit Cloud

### View Logs

1. Click "Manage app" (bottom right)
2. Click "Logs" tab
3. Look for errors in:
   - Package installation
   - App startup
   - Runtime errors

### Common Log Messages

**Success:**
```
You can now view your Streamlit app in your browser.
```

**Installing packages:**
```
Collecting streamlit
Successfully installed streamlit-x.x.x
```

**Error:**
```
ERROR: Could not find a version that satisfies the requirement
```

## 📦 Package Version Compatibility

### Tested Configurations

**Configuration 1 (Recommended):**
```txt
streamlit==1.28.0
pandas==2.0.3
numpy==1.24.3
scikit-learn==1.3.2
xgboost==2.0.3
joblib==1.3.0
plotly==5.17.0
```

**Configuration 2 (Latest):**
```txt
streamlit>=1.28.0
pandas>=2.0.0
numpy>=1.24.0
scikit-learn>=1.3.0
xgboost>=2.0.0
joblib>=1.3.0
plotly>=5.17.0
```

## 🔧 Advanced Troubleshooting

### Local Testing Before Deployment

```bash
# Create clean virtual environment
python -m venv test_env
test_env\Scripts\activate  # Windows
# source test_env/bin/activate  # Mac/Linux

# Install from requirements.txt
pip install -r requirements.txt

# Test app
streamlit run streamlit_app.py

# Check for errors
python -c "import streamlit, pandas, numpy, sklearn, xgboost, joblib, plotly"
```

### Verify Model Loading

```python
# test_model_loading.py
from pathlib import Path
import joblib

project_root = Path(__file__).parent
model_path = project_root / 'outputs' / 'models' / 'xgboost_model.pkl'

print(f"Model path: {model_path}")
print(f"Model exists: {model_path.exists()}")

if model_path.exists():
    model = joblib.load(model_path)
    print(f"Model loaded: {type(model)}")
    print(f"Model features: {model.n_features_in_}")
```

### Check File Sizes

```bash
# Check model file size
ls -lh outputs/models/xgboost_model.pkl

# If > 100MB, use Git LFS
git lfs install
git lfs track "*.pkl"
```

## 📞 Getting Help

### Streamlit Community

- **Forum:** https://discuss.streamlit.io
- **Docs:** https://docs.streamlit.io
- **GitHub:** https://github.com/streamlit/streamlit

### Quick Links

- Deployment docs: https://docs.streamlit.io/streamlit-community-cloud/deploy-your-app
- Troubleshooting: https://docs.streamlit.io/knowledge-base
- App config: https://docs.streamlit.io/library/advanced-features/configuration

## ✅ Verification Checklist

Before seeking help, verify:

- [ ] `requirements.txt` in root directory
- [ ] All packages listed correctly
- [ ] Model file exists and is committed
- [ ] No absolute file paths in code
- [ ] App runs locally without errors
- [ ] All files pushed to GitHub
- [ ] Deployment logs checked
- [ ] Error messages noted

## 🎯 Success Indicators

Your app is deployed successfully when:

✅ Streamlit Cloud shows "Your app is live!"
✅ App URL is accessible
✅ No error messages in logs
✅ Model loads correctly
✅ Predictions generate successfully
✅ All pages/tabs work properly

---

## 📝 Quick Fix Reference

| Issue | Quick Fix |
|-------|-----------|
| ModuleNotFoundError | Add package to `requirements.txt` |
| Model not found | Check file path, commit model file |
| Import error | Update package versions |
| Memory error | Optimize model, use caching |
| Slow performance | Add `@st.cache_resource` |
| CORS error | Update `.streamlit/config.toml` |
| Path error | Use `pathlib.Path` |

---

**Need immediate help?** Run locally first:
```bash
streamlit run streamlit_app.py
```

If it works locally but not on Streamlit Cloud, the issue is likely:
1. Missing file in repository
2. Package version incompatibility
3. Path differences (Windows vs Linux)

---

*Last Updated: December 24, 2025*
*Status: All issues resolved ✅*
