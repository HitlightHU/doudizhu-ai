@echo off
setlocal
cd /d "%~dp0"
python -m doudizhu.gui
if errorlevel 1 (
  echo.
  echo Failed to start. Activate the virtual environment and run:
  echo   pip install -r requirements.txt
  pause
)
endlocal
