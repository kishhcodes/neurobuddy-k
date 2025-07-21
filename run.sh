#!/bin/bash

# Script to run the Face Emotion & Attention Detection API

# Check if virtual environment exists
if [ ! -d "projenv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv projenv
fi

# Activate virtual environment
source projenv/bin/activate

# Install requirements if needed
pip install -r requirements.txt

# Run the application
echo "Starting Face Emotion & Attention API..."
uvicorn main:app --reload

# Deactivate on exit
trap "deactivate" EXIT
