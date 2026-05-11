@echo off
setlocal

echo =========================================
echo B-Roll Finder Setup ^& Launcher
echo =========================================

:: Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python 3.10+ is not installed or not in PATH.
    echo Please download and install Python from https://www.python.org/downloads/
    echo Make sure to check the box "Add Python to PATH" during installation.
    pause
    exit /b 1
)

:: Check if venv exists, create if not
if not exist venv (
    echo [INFO] Creating virtual environment...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Failed to create virtual environment.
        pause
        exit /b 1
    )
)

:: Activate venv
call venv\Scripts\activate.bat

:: Clear stale proxy environment variables
set HTTP_PROXY=
set HTTPS_PROXY=
set http_proxy=
set https_proxy=

:: Install requirements
echo [INFO] Installing requirements...
pip install -r requirements.txt --proxy=""
if %errorlevel% neq 0 (
    echo [ERROR] Failed to install requirements.
    pause
    exit /b 1
)

:: Run Streamlit
echo [INFO] Starting B-Roll Finder...
streamlit run app.py --server.port 8080

pause
