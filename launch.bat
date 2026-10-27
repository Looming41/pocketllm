@echo off
rem PocketLLM launcher (Windows). Runs from the drive it lives on.
cd /d "%~dp0"
set "POCKETLLM_ROOT=%CD%"
rem a portable Python on the drive (scripts/fetch_python.py) wins over the computer's own
set "PYEXE=%CD%\python\windows-x64\python.exe"
if exist "%PYEXE%" goto run
set "PYEXE=python"
where py >nul 2>nul && set "PYEXE=py"
:run
start "" http://127.0.0.1:8080
"%PYEXE%" -m pocketllm serve %*
if errorlevel 1 (
  echo.
  echo Python 3 is required on this computer: https://www.python.org/downloads/
  echo or put a portable Python on the drive: python scripts\fetch_python.py ^<drive^>\PocketLLM
  pause
)
