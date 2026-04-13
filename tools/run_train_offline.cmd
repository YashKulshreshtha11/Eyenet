@echo off
setlocal

set "HF_HUB_OFFLINE=1"
set "TRANSFORMERS_OFFLINE=1"

REM CPU speed knobs (safe on GPU too; ignored there).
REM Tune to your machine. For i5-1135G7: 8 logical processors.
if "%OMP_NUM_THREADS%"=="" set "OMP_NUM_THREADS=8"
if "%MKL_NUM_THREADS%"=="" set "MKL_NUM_THREADS=8"

cd /d "%~dp0.."

REM Usage:
REM   tools\run_train_offline.cmd [training_pipeline args...]

.\.venv311\Scripts\python.exe -u training_pipeline.py %*

