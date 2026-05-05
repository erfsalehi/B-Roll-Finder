#!/bin/bash

echo "========================================="
echo "B-Roll Finder Setup & Launcher"
echo "========================================="

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 could not be found."
    echo "Please install Python 3.10+ to run this application."
    exit 1
fi

# Check if venv exists, create if not
if [ ! -d "venv" ]; then
    echo "[INFO] Creating virtual environment..."
    python3 -m venv venv
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment."
        exit 1
    fi
fi

# Activate venv
source venv/bin/activate

# Install requirements
echo "[INFO] Installing requirements..."
pip install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to install requirements."
    exit 1
fi

# Run Streamlit
echo "[INFO] Starting B-Roll Finder..."
streamlit run app.py
