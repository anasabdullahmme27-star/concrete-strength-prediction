@echo off
title Concrete Strength Predictor
cd /d "%~dp0"

set "PY=.venv\Scripts\python.exe"

echo ============================================================
echo    Concrete Strength Predictor
echo ============================================================
echo.

if exist "%PY%" goto CHECKDEPS

echo [setup] No virtual environment found. Creating one...
py -3.13 -m venv .venv
if exist "%PY%" goto INSTALL
py -3.12 -m venv .venv
if exist "%PY%" goto INSTALL
py -3 -m venv .venv
if exist "%PY%" goto INSTALL
python -m venv .venv
if exist "%PY%" goto INSTALL

echo.
echo ERROR: Python could not be found on this machine.
echo Install Python 3.13 from https://www.python.org/downloads/
echo and make sure "Add python.exe to PATH" is ticked, then run this file again.
echo.
pause
exit /b 1

:CHECKDEPS
"%PY%" -c "import streamlit, sklearn, pandas, plotly, lightgbm, xgboost" >nul 2>&1
if errorlevel 1 goto INSTALL
goto RUN

:INSTALL
echo [setup] Installing dependencies. The first run takes a few minutes...
echo.
"%PY%" -m pip install --upgrade pip --quiet
"%PY%" -m pip install -r requirements.txt
if errorlevel 1 goto PIPFAIL
echo.
echo [setup] Dependencies installed.
goto RUN

:PIPFAIL
echo.
echo ERROR: Installing the dependencies failed. Read the messages above,
echo check your internet connection, and try again.
echo.
pause
exit /b 1

:RUN
echo.
echo Starting the application...
echo Your browser will open automatically at http://localhost:8502
echo.
echo Keep this window open while you use the app.
echo To stop the server: press Ctrl+C, or just close this window.
echo.
"%PY%" -m streamlit run streamlit_app.py --server.port 8502
if errorlevel 1 goto RUNFAIL
exit /b 0

:RUNFAIL
echo.
echo The application stopped with an error. See the messages above.
echo.
pause
exit /b 1
