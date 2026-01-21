@echo off
REM ============================================
REM FelicityTech AI Data Analyst Launcher
REM Created by Solomon Eniola Adegoke
REM ============================================

title FelicityTech AI Data Analyst

echo.
echo ================================================
echo   FelicityTech AI Data Analyst
echo   Created by Solomon Eniola Adegoke
echo ================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python is not installed or not in PATH
    echo.
    echo Please install Python from https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation
    echo.
    pause
    exit /b 1
)

echo [OK] Python found: 
python --version
echo.

REM Check if virtual environment exists
if exist "venv\Scripts\activate.bat" (
    echo [INFO] Activating virtual environment...
    call venv\Scripts\activate.bat
    echo [OK] Virtual environment activated
) else (
    echo [INFO] No virtual environment found
    echo [INFO] Using system Python
)
echo.

REM Check if requirements are installed
python -c "import streamlit" >nul 2>&1
if errorlevel 1 (
    echo [WARNING] Streamlit not found. Installing dependencies...
    echo.
    python -m pip install -r requirements.txt
    if errorlevel 1 (
        echo [ERROR] Failed to install dependencies
        echo.
        pause
        exit /b 1
    )
    echo [OK] Dependencies installed successfully
    echo.
)

REM Check if app.py exists
if not exist "app.py" (
    echo [ERROR] app.py not found in current directory
    echo Please make sure you're running this from the project folder
    echo.
    pause
    exit /b 1
)

REM Display helpful information
echo ================================================
echo  Starting the application...
echo ================================================
echo.
echo The app will open automatically in your browser
echo URL: http://localhost:8501
echo.
echo To stop the server: Press Ctrl+C in this window
echo.
echo ================================================
echo.

REM Start Streamlit
streamlit run app.py

REM If Streamlit exits
echo.
echo ================================================
echo  Application stopped
echo ================================================
echo.
pause