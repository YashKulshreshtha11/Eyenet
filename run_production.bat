@echo off
echo ==========================================================
echo       EyeNet Elite - Unified Production Launch
echo ==========================================================
echo.

:: 1. Build Frontend
echo [1/3] Building latest Frontend bundle...
cd frontend
call npm run build
cd ..

:: 2. Launch Backend (Served as a unified app)
echo.
echo [2/3] Starting Unified Server on http://localhost:8000
echo.
echo [TIP] This server now handles both API and UI.
echo       You do not need to run npm run dev anymore.
echo.

python -m uvicorn backend.app_server:app --host 0.0.0.0 --port 8000
