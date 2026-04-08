@echo off
REM ============================================================
REM Generative Kitting System - Streamlit UI Launcher (Windows)
REM ============================================================

set "PROJECT_DIR=%~dp0"
set "VENV_NAME=.aikido"
set "VENV_DIR=%PROJECT_DIR%%VENV_NAME%"
set "REQUIREMENTS=%PROJECT_DIR%requirements.txt"
set "STREAMLIT_APP=%PROJECT_DIR%ui\streamlit_app.py"

REM ── Step 1: Create virtual environment if missing ────────────
if exist "%VENV_DIR%\Scripts\activate.bat" goto :activate

echo [1/4] Virtual environment not found. Creating "%VENV_NAME%"...

REM Find Python — try py launcher first, then bare python
where py >nul 2>&1
if %errorlevel% equ 0 (
    set "PYTHON=py"
) else (
    where python >nul 2>&1
    if %errorlevel% equ 0 (
        set "PYTHON=python"
    ) else (
        echo [ERROR] Python not found. Install Python 3.9+ and make sure it is on PATH.
        pause
        exit /b 1
    )
)

%PYTHON% -m venv "%VENV_DIR%"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to create virtual environment at "%VENV_DIR%"
    pause
    exit /b 1
)
echo [1/4] Virtual environment created.
goto :activate

:activate
REM ── Step 2: Activate ─────────────────────────────────────────
echo [2/4] Activating virtual environment "%VENV_NAME%"...
call "%VENV_DIR%\Scripts\activate.bat"
if %errorlevel% neq 0 (
    echo [ERROR] Failed to activate virtual environment.
    pause
    exit /b 1
)

REM ── Step 3: Check / install dependencies ─────────────────────
echo [3/4] Checking dependencies...
if not exist "%REQUIREMENTS%" (
    echo [WARN] requirements.txt not found at "%REQUIREMENTS%". Skipping dependency install.
    goto :launch
)
pip install -r "%REQUIREMENTS%" --quiet
if %errorlevel% neq 0 (
    echo [WARN] Some dependencies failed to install. Continuing anyway...
)

REM ── Step 4: Launch Streamlit ─────────────────────────────────
:launch
echo [4/4] Launching Streamlit UI...
echo.
echo   App:   %STREAMLIT_APP%
echo   URL:   http://localhost:8501
echo   Stop:  Ctrl+C
echo.
call streamlit run "%STREAMLIT_APP%"
echo.
echo Streamlit exited.
pause
