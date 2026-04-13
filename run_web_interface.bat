@echo off
REM =========================================================================
REM The Analyst — Financial NLP Web Interface Startup Script
REM =========================================================================
REM This script starts both the FastAPI backend server and opens the frontend
REM in your browser.

setlocal enabledelayedexpansion

REM Color codes for output
for /F %%a in ('echo prompt $H ^| cmd') do set "BS=%%a"

echo.
echo ************************************************************
echo          The Analyst - Financial Orchestration
echo ************************************************************
echo.

REM Check if venv is activated
if not defined VIRTUAL_ENV (
    echo [INFO] Virtual environment not activated. Activating...
    call venv\Scripts\activate.bat
    if errorlevel 1 (
        echo [ERROR] Failed to activate virtual environment
        echo Please ensure venv exists. Create with: python -m venv venv
        pause
        exit /b 1
    )
)

echo [OK] Environment ready
echo.

REM Check if FastAPI is installed
python -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo [INFO] FastAPI not found. Installing dependencies...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        pause
        exit /b 1
    )
)

echo [OK] Dependencies installed
echo.

REM Check environment variables
if not defined OPENAI_API_KEY (
    echo [WARNING] OPENAI_API_KEY not set in environment
    echo Please ensure .env file exists with OPENAI_API_KEY
)

echo.
echo ************************************************************
echo          Starting Servers...
echo ************************************************************
echo.

REM Note: We use 'start' command to open in new windows
REM This allows the user to see both outputs simultaneously

REM Start FastAPI backend in new window
echo [1/2] Starting FastAPI backend on http://localhost:8000...
start "The Analyst - Backend" cmd /k "python web.py"

REM Give backend time to start
timeout /t 3 /nobreak

REM Start HTTP server for frontend in new window
echo [2/2] Starting frontend server on http://localhost:5000...
start "The Analyst - Frontend" cmd /k "python -m http.server 5000"

REM Give frontend time to start
timeout /t 2 /nobreak

REM Open browser
echo [3/3] Opening browser to http://localhost:5000...
timeout /t 1 /nobreak

REM Try to open in default browser using multiple methods
start "" http://localhost:5000

echo.
echo ************************************************************
echo          Ready!
echo ************************************************************
echo.
echo Backend (FastAPI):  http://localhost:8000
echo Frontend (UI):      http://localhost:5000
echo.
echo Both servers are running in separate windows above.
echo Close any window to stop the server.
echo.
echo API Documentation: http://localhost:8000/docs
echo.

pause
