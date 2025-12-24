@echo off
REM Alzheimer Risk Assessment - Streamlit Launcher
REM ===============================================

echo.
echo ===================================================
echo  Alzheimer's Disease Risk Assessment System
echo  Streamlit Application Launcher
echo ===================================================
echo.

REM Check if .venv exists
if exist ".venv\Scripts\python.exe" (
    echo [INFO] Using virtual environment (.venv)
    set PYTHON_CMD=.venv\Scripts\python.exe
    set PIP_CMD=.venv\Scripts\pip.exe
    set STREAMLIT_CMD=.venv\Scripts\streamlit.exe
) else if exist "venv\Scripts\python.exe" (
    echo [INFO] Using virtual environment (venv)
    set PYTHON_CMD=venv\Scripts\python.exe
    set PIP_CMD=venv\Scripts\pip.exe
    set STREAMLIT_CMD=venv\Scripts\streamlit.exe
) else (
    REM Try to find python in PATH
    python --version >nul 2>&1
    if errorlevel 1 (
        echo [ERROR] Python is not installed or not in PATH
        echo Please install Python 3.8 or higher from python.org
        pause
        exit /b 1
    )
    set PYTHON_CMD=python
    set PIP_CMD=pip
    set STREAMLIT_CMD=streamlit
)

echo [INFO] Python found
echo.

REM Check if dependencies are installed
echo [INFO] Checking dependencies...
%PYTHON_CMD% -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Streamlit not found
    echo [INFO] Installing dependencies...
    %PIP_CMD% install -r requirements_streamlit.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
) else (
    echo [INFO] Dependencies OK
)

echo.
echo [INFO] Verifying setup...
%PYTHON_CMD% verify_streamlit_setup.py

echo.
echo ===================================================
echo  Starting Streamlit Application...
echo ===================================================
echo.
echo The app will open in your default web browser at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the server
echo.

REM Run Streamlit
%STREAMLIT_CMD% run streamlit_app.py

pause
