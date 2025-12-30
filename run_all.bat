@echo off
REM Travel ML Project - Windows Setup Script
REM Run this script to set up and start the project

echo ============================================================
echo           TRAVEL ML PROJECT SETUP (Windows)
echo           MLOps Capstone Project
echo ============================================================

REM Check Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo ERROR: Python is not installed or not in PATH
    pause
    exit /b 1
)

echo.
echo [1/6] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate

echo.
echo [2/6] Installing dependencies...
pip install -r requirements.txt

echo.
echo [3/6] Generating datasets...
cd data
python generate_data.py
cd ..

echo.
echo [4/6] Training Flight Price Model...
cd models
python flight_price_model.py

echo.
echo [5/6] Training Gender Classifier...
python gender_classifier.py

echo.
echo [6/6] Training Hotel Recommender...
python hotel_recommender.py
cd ..

echo.
echo ============================================================
echo           SETUP COMPLETE!
echo ============================================================
echo.
echo To start the API:
echo   cd api ^&^& python app.py
echo.
echo To start the Dashboard:
echo   cd streamlit ^&^& streamlit run app.py
echo.
echo To run tests:
echo   pytest tests/ -v
echo.
pause
