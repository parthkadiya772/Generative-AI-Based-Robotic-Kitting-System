#!/usr/bin/env bash
# ============================================================
# Generative Kitting System — Streamlit UI Launcher (Linux/Mac)
# ============================================================

PROJECT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV_NAME=".aikido"
VENV_DIR="$PROJECT_DIR/$VENV_NAME"
PYTHON_EXE="$VENV_DIR/bin/python"
PIP_EXE="$VENV_DIR/bin/pip"
REQUIREMENTS="$PROJECT_DIR/requirements.txt"
STREAMLIT_APP="$PROJECT_DIR/ui/streamlit_app.py"

# Flag to track if we need to install dependencies
NEEDS_INSTALL=false

catch_error() {
    echo ""
    echo "============================================================"
    echo "[ERROR] Script failed. Press Enter to close this window..."
    echo "============================================================"
    read -r
    exit 1
}

echo "============================================================"
echo " Generative Kitting - Streamlit UI"
echo "============================================================"
echo " Project : $PROJECT_DIR"
echo " Venv    : $VENV_DIR"
echo "============================================================"
echo ""

# ── Step 1: Create virtual environment if missing ──────────
if [ -d "$VENV_DIR" ] && [ -f "$PYTHON_EXE" ]; then
    echo "[1/3] Virtual environment found. Skipping setup..."
else
    echo "[1/3] Virtual environment not found or incomplete at \"$VENV_DIR\""
    echo "      Creating it now..."
    NEEDS_INSTALL=true # Mark that we need to install packages after this

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
        catch_error
    fi

    echo "      Using: $SYS_PYTHON"
    $SYS_PYTHON -m venv "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo ""
        echo "[ERROR] Failed to create virtual environment."
        catch_error
    fi
    echo "      Virtual environment created successfully."
fi

# ── Step 2: Install dependencies ONLY IF the venv was just created ──
if [ "$NEEDS_INSTALL" = true ]; then
    echo "[2/3] First-time setup: Installing dependencies..."
    
    # Upgrade pip first
    "$PYTHON_EXE" -m pip install --upgrade pip --quiet 2>/dev/null

    if [ ! -f "$REQUIREMENTS" ]; then
        echo "[WARN] requirements.txt not found — skipping dependency install."
    else
        echo "      Processing packages line-by-line..."
        while read -r line || [ -n "$line" ]; do
            cleaned_line=$(echo "$line" | tr -d '\r' | xargs)
            [[ -z "$cleaned_line" || "$cleaned_line" == \#* ]] && continue
            
            echo "      -> Installing $cleaned_line..."
            if ! "$PIP_EXE" install "$cleaned_line" --quiet --no-cache-dir 2>/dev/null; then
                echo "      [SKIPPED] Non-compatible or failing package: $cleaned_line"
            fi
        done < "$REQUIREMENTS"
    fi
else
    echo "[2/3] Skipping dependency check (Environment already configured)."
fi

# Activate the environment for any child processes launched by this script.
source "$VENV_DIR/bin/activate"

# ── Step 3: Launch Streamlit ───────────────────────────────
if [ ! -f "$STREAMLIT_APP" ]; then
    echo ""
    echo "[ERROR] App file not found: $STREAMLIT_APP"
    catch_error
fi

echo "[3/3] Launching Streamlit UI..."
echo ""
echo "  App : $STREAMLIT_APP"
echo "  URL : http://localhost:8501"
echo "  Stop: Ctrl+C"
echo ""

python -m streamlit run "$STREAMLIT_APP"
echo ""
echo "[INFO] Streamlit process exited (code: $?)."
