@echo off
setlocal ENABLEEXTENSIONS

REM Always run using the project-local venv (avoids Microsoft Store Python issues).
set "VENV_PY=%~dp0.venv311\Scripts\python.exe"

if not exist "%VENV_PY%" (
  echo [ERROR] venv Python not found: "%VENV_PY%"
  echo         Create it with: python -m venv .venv311
  exit /b 1
)

if "%~1"=="" (
  echo Usage:
  echo   run_venv.bat server
  echo   run_venv.bat train [training_pipeline args...]
  echo.
  echo Examples:
  echo   run_venv.bat server
  echo   run_venv.bat train --data_dir "..\use_this\archive\dataset_split_with_odir" --robust_aug
  exit /b 0
)

if /I "%~1"=="server" (
  cd /d "%~dp0"
  "%VENV_PY%" -m uvicorn backend.app_server:app --host 0.0.0.0 --port 8000
  exit /b %ERRORLEVEL%
)

if /I "%~1"=="train" (
  cd /d "%~dp0"
  REM Robustly forward all args AFTER the word "train".
  set "REST_ARGS="
  for /f "tokens=1,* delims= " %%A in ("%*") do set "REST_ARGS=%%B"
  call "%VENV_PY%" training_pipeline.py %REST_ARGS%
  exit /b %ERRORLEVEL%
)

echo [ERROR] Unknown command: %~1
echo Use: run_venv.bat server ^| train
exit /b 2

