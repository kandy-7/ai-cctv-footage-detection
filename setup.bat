@echo off
echo ========================================
echo AI CCTV Analysis System - Setup
echo ========================================
echo.

echo [1/4] Setting up Python virtual environment...
cd backend
if exist "venv\" (
    echo Virtual environment already exists, skipping...
) else (
    python -m venv venv
    echo Virtual environment created successfully!
)

echo.
echo [2/4] Installing Python dependencies...
call venv\Scripts\activate
pip install -r requirements.txt

cd ..

echo.
echo [3/4] Setting up Node.js frontend...
cd frontend

echo.
echo [4/4] Installing Node dependencies...
call npm install

cd ..

echo.
echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo To start the application, run: start.bat
echo.
pause
