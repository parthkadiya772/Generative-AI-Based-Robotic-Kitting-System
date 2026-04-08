#!/usr/bin/env bash
# ============================================================
# Generative Kitting System — Streamlit UI Launcher (Linux/Mac)
# ============================================================

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_NAME=".aikido"
VENV_DIR="$PROJECT_DIR/$VENV_NAME"
REQUIREMENTS="$PROJECT_DIR/requirements.txt"
STREAMLIT_APP="$PROJECT_DIR/ui/streamlit_app.py"

# ── Step 1: Create virtual environment if missing ──────────
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[1/4] Virtual environment not found. Creating \"$VENV_NAME\"..."

    # Find Python — prefer python3, fall back to python
    if command -v python3 &>/dev/null; then
        PYTHON=python3
    elif command -v python &>/dev/null; then
        PYTHON=python
    else
        echo "[ERROR] Python not found. Install Python 3.9+ and make sure it is on PATH."
        exit 1
    fi

    $PYTHON -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtual environment at \"$VENV_DIR\""
        exit 1
    fi
    echo "[1/4] Virtual environment created."
else
    echo "[1/4] Virtual environment found."
fi

# ── Step 2: Activate ───────────────────────────────────────
echo "[2/4] Activating virtual environment \"$VENV_NAME\"..."
source "$VENV_DIR/bin/activate"
if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to activate virtual environment."
    exit 1
fi

# ── Step 3: Check / install dependencies ───────────────────
echo "[3/4] Checking dependencies..."
if [ ! -f "$REQUIREMENTS" ]; then
    echo "[WARN] requirements.txt not found at \"$REQUIREMENTS\". Skipping dependency install."
else
    pip install -r "$REQUIREMENTS" --quiet 2>&1 || {
        echo "[WARN] Some dependencies failed to install. Continuing anyway..."
    }
fi

# ── Step 4: Launch Streamlit ───────────────────────────────
echo "[4/4] Launching Streamlit UI..."
echo ""
echo "  App:   $STREAMLIT_APP"
echo "  URL:   http://localhost:8501"
echo "  Stop:  Ctrl+C"
echo ""
streamlit run "$STREAMLIT_APP"
