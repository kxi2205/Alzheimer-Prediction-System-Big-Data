# Streamlit Deployment Guide

## 🚀 Running the Alzheimer's Risk Assessment App with Streamlit

This guide will help you deploy and run the Alzheimer's Disease Risk Assessment application using Streamlit.

## Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- The trained XGBoost model file in `outputs/models/xgboost_model.pkl`

## Installation

### 1. Install Dependencies

```bash
pip install -r requirements_streamlit.txt
```

### 2. Verify Model Files

Ensure the following files exist:
- `outputs/models/xgboost_model.pkl` - The trained model
- `outputs/config/final_features.json` - Feature configuration (optional)

## Running Locally

### Start the Application

```bash
streamlit run streamlit_app.py
```

The application will automatically open in your default web browser at `http://localhost:8501`

### Alternative Port

If port 8501 is already in use:

```bash
streamlit run streamlit_app.py --server.port 8502
```

## Deployment Options

### 1. Streamlit Community Cloud (Recommended)

1. **Push to GitHub**:
   ```bash
   git add .
   git commit -m "Add Streamlit app"
   git push origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with GitHub
   - Click "New app"
   - Select your repository
   - Set main file path: `streamlit_app.py`
   - Click "Deploy"

3. **Important**: Ensure model files are included in your repository or use Git LFS for large files.

### 2. Docker Deployment

Create a `Dockerfile`:

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY requirements_streamlit.txt .
RUN pip install --no-cache-dir -r requirements_streamlit.txt

COPY . .

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Build and run:

```bash
docker build -t alzheimer-app .
docker run -p 8501:8501 alzheimer-app
```

### 3. Heroku Deployment

Create `setup.sh`:

```bash
mkdir -p ~/.streamlit/

echo "\
[server]\n\
headless = true\n\
port = $PORT\n\
enableCORS = false\n\
\n\
" > ~/.streamlit/config.toml
```

Create `Procfile`:

```
web: sh setup.sh && streamlit run streamlit_app.py
```

Deploy:

```bash
heroku create your-app-name
git push heroku main
```

### 4. AWS EC2 / Azure / GCP

1. **Set up a virtual machine**
2. **Install Python and dependencies**
3. **Clone your repository**
4. **Run with tmux or screen**:
   ```bash
   tmux new -s streamlit
   streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
   ```
5. **Configure firewall** to allow port 8501
6. **Optional**: Set up nginx as reverse proxy

## Application Features

### 📋 Assessment Sections

1. **Personal Information**: Age, gender, ethnicity, education
2. **Physical Health**: BMI, blood pressure, cholesterol levels
3. **Lifestyle Factors**: Smoking, alcohol, exercise, diet, sleep
4. **Medical History**: Family history, chronic conditions
5. **Cognitive Assessment**: MMSE scores, cognitive symptoms

### 📊 Results

- **Risk Score**: 0-10 scale with visual gauge
- **Risk Category**: Low, Moderate, or High
- **Clinical Adjustments**: Explanation of risk factors
- **Personalized Recommendations**: Health and lifestyle suggestions
- **Downloadable Report**: PDF/text summary of assessment

## Troubleshooting

### Model Not Found Error

If you see "Model not found" error:

1. Check if `outputs/models/xgboost_model.pkl` exists
2. Run the model training notebooks to generate the model
3. Verify the model path in `streamlit_app.py`

### Import Errors

```bash
pip install --upgrade -r requirements_streamlit.txt
```

### Port Already in Use

```bash
# Find process using port 8501 (Windows)
netstat -ano | findstr :8501

# Find process using port 8501 (Linux/Mac)
lsof -i :8501

# Use different port
streamlit run streamlit_app.py --server.port 8502
```

### Memory Issues

For large models, increase memory:

```bash
streamlit run streamlit_app.py --server.maxUploadSize 500
```

## Configuration

### Customize Theme

Edit `.streamlit/config.toml`:

```toml
[theme]
primaryColor="#10b981"
backgroundColor="#ffffff"
secondaryBackgroundColor="#f9fafb"
textColor="#1f2937"
```

### Enable Analytics

```toml
[browser]
gatherUsageStats = true
```

## Security Considerations

- **Never expose** patient health information
- Use **HTTPS** in production
- Implement **authentication** if needed
- Add **rate limiting** for API calls
- **Sanitize** user inputs

## Performance Optimization

1. **Cache model loading**:
   ```python
   @st.cache_resource
   def load_model():
       return joblib.load('model.pkl')
   ```

2. **Use session state** for form data
3. **Optimize imports** - only import what's needed
4. **Compress model files** if large

## Monitoring

### Local Logs

```bash
streamlit run streamlit_app.py --logger.level=debug
```

### Production Monitoring

- Use Streamlit Cloud built-in analytics
- Integrate with Google Analytics
- Set up error tracking (e.g., Sentry)

## Support

For issues or questions:
- Check [Streamlit Documentation](https://docs.streamlit.io)
- Visit [Streamlit Community Forum](https://discuss.streamlit.io)
- Review application logs

## License

This application is for educational and research purposes only.

---

**⚠️ Medical Disclaimer**: This tool is not intended for clinical diagnosis. Always consult healthcare professionals for medical advice.
