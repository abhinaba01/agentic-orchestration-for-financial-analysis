#!/bin/bash

# =========================================================================
# The Analyst — Financial NLP Web Interface Startup Script
# =========================================================================
# This script starts both the FastAPI backend server and opens the frontend
# in your browser.

set -e

echo ""
echo "************************************************************"
echo "          The Analyst - Financial Orchestration"
echo "************************************************************"
echo ""

# Check if venv is activated
if [[ -z "${VIRTUAL_ENV}" ]]; then
    echo "[INFO] Virtual environment not activated. Activating..."
    source venv/bin/activate || {
        echo "[ERROR] Failed to activate virtual environment"
        echo "Please ensure venv exists. Create with: python -m venv venv"
        exit 1
    }
fi

echo "[OK] Environment ready"
echo ""

# Check if FastAPI is installed
python -c "import fastapi" 2>/dev/null || {
    echo "[INFO] FastAPI not found. Installing dependencies..."
    pip install -r requirements.txt
}

echo "[OK] Dependencies installed"
echo ""

# Check environment variables
if [[ -z "${OPENAI_API_KEY}" ]]; then
    if [[ ! -f .env ]]; then
        echo "[WARNING] OPENAI_API_KEY not set and .env file not found"
        echo "Please ensure .env file exists with OPENAI_API_KEY"
    else
        echo "[INFO] Loading .env file..."
        export $(cat .env | grep -v '#' | xargs)
    fi
fi

echo ""
echo "************************************************************"
echo "          Starting Servers..."
echo "************************************************************"
echo ""

# Function to cleanup background processes on exit
cleanup() {
    echo ""
    echo "Shutting down servers..."
    kill $BACKEND_PID 2>/dev/null || true
    kill $FRONTEND_PID 2>/dev/null || true
    wait $BACKEND_PID 2>/dev/null || true
    wait $FRONTEND_PID 2>/dev/null || true
    echo "Shut down complete."
}

trap cleanup EXIT

# Start FastAPI backend in background
echo "[1/2] Starting FastAPI backend on http://localhost:8000..."
python web.py &
BACKEND_PID=$!

# Wait for backend to start
sleep 3

# Start HTTP server for frontend in background
echo "[2/2] Starting frontend server on http://localhost:5000..."
python -m http.server 5000 &
FRONTEND_PID=$!

# Wait for frontend to start
sleep 2

# Open browser
echo "[3/3] Opening browser to http://localhost:5000..."

# Try to open in default browser (works on macOS and Linux)
if command -v xdg-open &> /dev/null; then
    xdg-open http://localhost:5000  # Linux
elif command -v open &> /dev/null; then
    open http://localhost:5000  # macOS
else
    echo "[INFO] Please open http://localhost:5000 in your browser"
fi

echo ""
echo "************************************************************"
echo "          Ready!"
echo "************************************************************"
echo ""
echo "Backend (FastAPI):  http://localhost:8000"
echo "Frontend (UI):      http://localhost:5000"
echo ""
echo "API Documentation: http://localhost:8000/docs"
echo ""
echo "Press Ctrl+C to stop the servers..."
echo ""

# Keep the script running
wait
