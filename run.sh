#!/usr/bin/env bash
# ══════════════════════════════════════════════════════════════════════════
#  EyeNet – One-Command Launcher (Linux / macOS)
# ══════════════════════════════════════════════════════════════════════════

set -e
cd "$(dirname "$0")"

echo ""
echo "  ========================================================"
echo "   EyeNet - Retinal Disease Detection & Analysis System"
echo "  ========================================================"
echo ""

# ── Install dependencies ─────────────────────────────────────────────────
echo "[1/3] Installing Python dependencies..."
pip install -r requirements.txt --quiet

# ── Create weights directory ─────────────────────────────────────────────
mkdir -p weights

# ── Start backend ────────────────────────────────────────────────────────
echo "[2/3] Starting EyeNet backend (FastAPI)..."
python -m uvicorn backend.app_server:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!

sleep 2

# ── Open frontend ────────────────────────────────────────────────────────
echo "[3/3] Opening EyeNet frontend..."
FRONTEND="http://localhost:8000"
if [[ "$OSTYPE" == "darwin"* ]]; then
  open "$FRONTEND"
elif command -v xdg-open &>/dev/null; then
  xdg-open "$FRONTEND"
else
  echo "  Please open manually: $FRONTEND"
fi

echo ""
echo "  EyeNet is running!"
echo "   Backend API : http://localhost:8000"
echo "   API Docs    : http://localhost:8000/docs"
echo "   Frontend    : $FRONTEND"
echo ""
echo "  Press Ctrl+C to stop."

wait $BACKEND_PID
