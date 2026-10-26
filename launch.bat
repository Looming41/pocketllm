@echo off
rem PocketLLM launcher (Windows). Runs from the drive it lives on.
cd /d "%~dp0"
set POCKETLLM_ROOT=%CD%
where py >nul 2>nul && (set PY=py -3) || (set PY=python)
start "" http://127.0.0.1:8080
%PY% -m pocketllm serve %*
if errorlevel 1 (
  echo.
  echo Python 3 is required on this computer: https://www.python.org/downloads/
  pause
)
