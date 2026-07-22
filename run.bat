@echo off
REM ============================================
REM  AgentResearch — Launch Script
REM  Multi-Agent AI Research Platform
REM ============================================

echo.
echo ========================================
echo   AgentResearch - Multi-Agent AI Research
echo ========================================
echo.

REM Check for Python
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3.11+
    pause
    exit /b 1
)

REM Check for .env
if not exist .env (
    echo [WARNING] .env file not found. Copying .env.example...
    copy .env.example .env >nul
    echo [INFO] Please edit .env with your API keys before running.
    echo.
)

REM Install backend dependencies
echo [1/2] Installing backend dependencies...
cd backend
pip install -r requirements.txt -q
cd ..

REM Install frontend dependencies
echo [2/2] Installing frontend dependencies...
cd frontend
pip install -r requirements.txt -q
cd ..

echo.
echo ========================================
echo   Starting AgentResearch...
echo ========================================
echo.

REM Launch Streamlit
cd frontend
streamlit run app.py --server.port 8501 --server.headless false
cd ..

pause
