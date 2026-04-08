#!/usr/bin/env bash
# ============================================================
# Generative Kitting System — Streamlit UI Launcher (Linux/Mac)
# ============================================================
#  Uses direct venv paths (venv/bin/python, venv/bin/pip) instead
#  of sourcing activate — avoids hardcoded-path issues in the
#  auto-generated activation script.
# ============================================================

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_NAME=".aikido"
VENV_DIR="$PROJECT_DIR/$VENV_NAME"
PYTHON_EXE="$VENV_DIR/bin/python"
PIP_EXE="$VENV_DIR/bin/pip"
REQUIREMENTS="$PROJECT_DIR/requirements.txt"
STREAMLIT_APP="$PROJECT_DIR/ui/streamlit_app.py"

echo "============================================================"
echo " Generative Kitting - Streamlit UI"
echo "============================================================"
echo " Project : $PROJECT_DIR"
echo " Venv    : $VENV_DIR"
echo "============================================================"
echo ""

# ── Step 1: Create virtual environment if missing ──────────
if [ -f "$PYTHON_EXE" ]; then
    echo "[1/4] Virtual environment found."
else
    echo "[1/4] Virtual environment not found at \"$VENV_DIR\""
    echo "      Creating it now..."

    # Find a system Python
    SYS_PYTHON=""
    for candidate in python3 python python3.11 python3.10 python3.9; do
        if command -v "$candidate" &>/dev/null; then
            SYS_PYTHON="$candidate"
            break
        fi
    done

    if [ -z "$SYS_PYTHON" ]; then
        echo ""
        echo "[ERROR] No Python interpreter found on PATH."
        echo "        Install Python 3.9+ and re-run."
        exit 1
    fi

    echo "      Using: $SYS_PYTHON"
    $SYS_PYTHON -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo ""
        echo "[ERROR] Failed to create virtual environment."
        exit 1
    fi
    echo "[1/4] Virtual environment created."
fi

# ── Step 2: Upgrade pip ─────────────────────────────────────
echo "[2/4] Checking pip..."
"$PYTHON_EXE" -m pip install --upgrade pip --quiet 2>/dev/null || \
    echo "[WARN] pip upgrade failed — continuing with existing version."

# ── Step 3: Install / verify dependencies ───────────────────
echo "[3/4] Checking dependencies..."
if [ ! -f "$REQUIREMENTS" ]; then
    echo "[WARN] requirements.txt not found — skipping dependency install."
else
    "$PIP_EXE" install -r "$REQUIREMENTS" --quiet 2>&1 || \
        echo "[WARN] Some packages failed to install — attempting to launch anyway..."
fi

# ── Step 4: Launch Streamlit ───────────────────────────────
if [ ! -f "$STREAMLIT_APP" ]; then
    echo ""
    echo "[ERROR] App file not found: $STREAMLIT_APP"
    exit 1
fi

echo "[4/4] Launching Streamlit UI..."
echo ""
echo "  App : $STREAMLIT_APP"
echo "  URL : http://localhost:8501"
echo "  Stop: Ctrl+C"
echo ""

"$PYTHON_EXE" -m streamlit run "$STREAMLIT_APP"
echo ""
echo "[INFO] Streamlit process exited (code: $?)."
