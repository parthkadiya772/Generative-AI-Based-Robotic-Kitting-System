#!/usr/bin/env bash
# ============================================================
# Generative Kitting System — Streamlit UI Launcher (Linux/Mac)
# ============================================================

set -e

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_DIR="$PROJECT_DIR/.aikido"
REQUIREMENTS="$PROJECT_DIR/requirements.txt"
STREAMLIT_APP="$PROJECT_DIR/ui/streamlit_app.py"

# ── Step 1: Check virtual environment ──────────────────────
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo "[ERROR] Virtual environment not found at $VENV_DIR"
    echo "Run:  python3 -m venv \"$VENV_DIR\""
    exit 1
fi

echo "[1/3] Activating virtual environment..."
source "$VENV_DIR/bin/activate"

# ── Step 2: Check / install dependencies ───────────────────
echo "[2/3] Checking dependencies..."
pip install -r "$REQUIREMENTS" --quiet 2>/dev/null || {
    echo "[WARN] Some dependencies failed to install. Continuing anyway..."
}

# ── Step 3: Launch Streamlit ───────────────────────────────
echo "[3/3] Launching Streamlit UI..."
echo ""
echo "  URL will open at http://localhost:8501"
echo "  Press Ctrl+C to stop."
echo ""
streamlit run "$STREAMLIT_APP"
