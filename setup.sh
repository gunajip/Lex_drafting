#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  LexDraft Assist — one-shot setup script
#  Usage:  chmod +x setup.sh && ./setup.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

VENV_DIR=".venv"
PYTHON="${PYTHON:-python3}"

echo ""
echo "╔══════════════════════════════════════════════════╗"
echo "║        LexDraft Assist — Setup Script            ║"
echo "╚══════════════════════════════════════════════════╝"
echo ""

# 1. Check Python ≥ 3.10
PY_VER=$($PYTHON -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$($PYTHON -c "import sys; print(sys.version_info.major)")
PY_MINOR=$($PYTHON -c "import sys; print(sys.version_info.minor)")
echo "► Python version: $PY_VER"
if [ "$PY_MAJOR" -lt 3 ] || { [ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]; }; then
  echo "  ERROR: Python 3.10+ is required."
  exit 1
fi

# 2. Create virtual environment
if [ ! -d "$VENV_DIR" ]; then
  echo "► Creating virtual environment in $VENV_DIR …"
  $PYTHON -m venv "$VENV_DIR"
else
  echo "► Virtual environment already exists."
fi

# Activate
source "$VENV_DIR/bin/activate"
echo "► Activated: $VENV_DIR"

# 3. Install dependencies
echo "► Installing Python dependencies…"
pip install --upgrade pip --quiet
pip install -r requirements.txt

# 4. Create required directories
mkdir -p parsed_historical uploads chroma_db
echo "► Created directories: parsed_historical/ uploads/ chroma_db/"

# 5. Check Ollama
if command -v ollama &>/dev/null; then
  echo "► Ollama found: $(ollama --version 2>&1 | head -1)"
  echo "  Available models:"
  ollama list 2>/dev/null | tail -n +2 | awk '{print "    •", $1}' || echo "    (none — pull a model with: ollama pull gemma3:4b)"
else
  echo "  WARNING: ollama not found in PATH."
  echo "  Install from: https://ollama.com"
  echo "  Then run:     ollama pull gemma3:4b"
fi

echo ""
echo "✅  Setup complete!"
echo ""
echo "  To start the server:"
echo "    source $VENV_DIR/bin/activate"
echo "    python server.py"
echo ""
echo "  Then open:  http://127.0.0.1:8000"
echo ""
