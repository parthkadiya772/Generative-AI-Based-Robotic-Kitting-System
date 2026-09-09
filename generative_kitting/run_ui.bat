@echo off
REM ============================================================
REM Generative Kitting System - Streamlit UI Launcher (Windows)
REM ============================================================

set "PROJECT_DIR=%~dp0"
set "VENV_DIR=%PROJECT_DIR%.aikido"
set "REQUIREMENTS=%PROJECT_DIR%requirements.txt"
set "STREAMLIT_APP=%PROJECT_DIR%ui\streamlit_app.py"

REM -- Step 1: Check virtual environment -----------------------
echo [1/3] Checking virtual environment...
if exist "%VENV_DIR%\Scripts\activate.bat" goto :activate_venv

echo Virtual environment not found. Creating it now...
where python >nul 2>nul
if errorlevel 1 (
	echo [ERROR] Python was not found on PATH.
	pause
	exit /b 1
)
python -m venv "%VENV_DIR%"
if errorlevel 1 (
	echo [ERROR] Failed to create the virtual environment.
	pause
	exit /b 1
)

:activate_venv
echo Activating virtual environment...
call "%VENV_DIR%\Scripts\activate.bat"

REM -- Step 2: Check / install dependencies --------------------
echo [2/3] Checking dependencies...
call python -m pip install -r "%REQUIREMENTS%" --quiet
if %errorlevel% neq 0 echo [WARN] Some dependencies failed to install. Continuing anyway...

REM -- Step 3: Launch Streamlit --------------------------------
echo [3/3] Launching Streamlit UI...
echo.
echo   URL will open at http://localhost:8501
echo   Press Ctrl+C to stop.
echo.
call streamlit run "%STREAMLIT_APP%"
echo.
echo Streamlit exited.
pause
goto :eof
