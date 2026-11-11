@echo off
setlocal
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
  echo Satyameter environment not found.
  echo First run PowerShell: Set-ExecutionPolicy -Scope Process Bypass
  echo Then run: .\setup_windows.ps1
  pause
  exit /b 1
)
.venv\Scripts\python.exe -m streamlit run app\streamlit_app.py
if errorlevel 1 (
  echo.
  echo Satyameter did not start. Review the error message above.
  pause
)
endlocal
