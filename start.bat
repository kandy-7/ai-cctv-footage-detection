@echo off
echo ========================================
echo AI CCTV Analysis System - Startup
echo ========================================
echo.

REM Check if backend virtual environment exists
if not exist "backend\venv\" (
    echo [ERROR] Virtual environment not found!
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Check if node_modules exists
if not exist "frontend\node_modules\" (
    echo [ERROR] Node modules not found! 
    echo Please run setup.bat first
    pause
    exit /b 1
)

echo Starting Backend Server...
start "Backend - FastAPI" cmd /k "cd backend && venv\Scripts\activate && python -m uvicorn main:app --reload --host 0.0.0.0 --port 8001"

timeout /t 3 /nobreak > nul

echo Starting Frontend Server...
start "Frontend - Vite" cmd /k "cd frontend && npm run dev"

echo.
echo ========================================
echo Both servers are starting...
echo Backend: http://localhost:8001
echo Frontend: http://localhost:5173
echo ========================================
echo.
echo Press any key to close this window...
pause > nul
