@echo off
REM ══════════════════════════════════════════════════════════════════════
REM  EyeNet – One-Command Launcher (Windows)
REM ══════════════════════════════════════════════════════════════════════

echo.
echo  ========================================================
echo   EyeNet - Retinal Disease Detection ^& Analysis System
echo  ========================================================
echo.

REM ── Install dependencies ───────────────────────────────────────────────
echo [1/3] Checking environment...
if exist "%~dp0env\Scripts\python.exe" (
    set PY_EXE=%~dp0env\Scripts\python.exe
    echo Using project environment at %~dp0env
) else if exist "%~dp0venv\Scripts\python.exe" (
    set PY_EXE=%~dp0venv\Scripts\python.exe
    echo Using project environment at %~dp0venv
) else if exist "D:\eye_env\Scripts\python.exe" (
    set PY_EXE=D:\eye_env\Scripts\python.exe
    echo Using environment at D:\eye_env
) else (
    set PY_EXE=python
    echo WARNING: No virtual environment found, using system python.
)

REM ── Create weights directory ───────────────────────────────────────────
if not exist "weights" mkdir weights

REM ── Start backend ──────────────────────────────────────────────────────
echo [2/3] Starting EyeNet backend (FastAPI)...
start "EyeNet Backend" cmd /k "cd /d %~dp0 && %PY_EXE% -m uvicorn backend.app_server:app --host 0.0.0.0 --port 8000 --reload"

REM Give backend a moment to boot
timeout /t 3 /nobreak >nul

REM ── Launch frontend ────────────────────────────────────────────────────
echo [3/3] Opening EyeNet frontend in browser...
start "" "http://localhost:8000"

echo.
echo  EyeNet is running!
echo   Backend API : http://localhost:8000
echo   API Docs    : http://localhost:8000/docs
echo   Frontend    : http://localhost:8000
echo.
echo  Press any key in the Backend window to stop the server.
echo.
