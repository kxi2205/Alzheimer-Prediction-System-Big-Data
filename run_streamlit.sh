#!/bin/bash
# Alzheimer Risk Assessment - Streamlit Launcher (Unix/Linux/Mac)
# ================================================================

echo ""
echo "==================================================="
echo " Alzheimer's Disease Risk Assessment System"
echo " Streamlit Application Launcher"
echo "==================================================="
echo ""

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 is not installed"
    echo "Please install Python 3.8 or higher"
    exit 1
fi

echo "[INFO] Python found: $(python3 --version)"
echo ""

# Check if virtual environment exists
if [ -d "venv" ]; then
    echo "[INFO] Activating virtual environment..."
    source venv/bin/activate
else
    echo "[WARNING] No virtual environment found"
    echo "[INFO] Consider creating one with: python3 -m venv venv"
    echo ""
fi

# Check if dependencies are installed
echo "[INFO] Checking dependencies..."
python3 -c "import streamlit" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[WARNING] Streamlit not found"
    echo "[INFO] Installing dependencies..."
    pip3 install -r requirements_streamlit.txt
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to install dependencies"
        exit 1
    fi
else
    echo "[INFO] Dependencies OK"
fi

echo ""
echo "[INFO] Verifying setup..."
python3 verify_streamlit_setup.py

echo ""
echo "==================================================="
echo " Starting Streamlit Application..."
echo "==================================================="
echo ""
echo "The app will open in your default web browser at:"
echo "http://localhost:8501"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

# Run Streamlit
streamlit run streamlit_app.py
