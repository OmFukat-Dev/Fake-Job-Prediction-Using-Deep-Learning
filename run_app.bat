@echo off
echo.
echo Fake Job Detector Launcher
echo ==============================
echo.

:: Change to project directory (directory of this script)
cd /d "%~dp0"

:: Check if directory exists
if not exist "app.py" (
    echo ERROR: app.py not found!
    pause
    exit /b 1
)

:: Resolve Python from virtual environment
echo Activating Python environment...
set "PY_EXE="
if exist ".venv\Scripts\python.exe" (
    set "PY_EXE=.venv\Scripts\python.exe"
    echo Virtual environment detected: .venv
) else if exist "venv\Scripts\python.exe" (
    set "PY_EXE=venv\Scripts\python.exe"
    echo Virtual environment detected: venv
) else (
    echo ERROR: Virtual environment not found. Create .venv or venv first.
    pause
    exit /b 1
)

:: Check and create processed data if needed
if not exist "data\processed\train_data.csv" (
    echo Running data cleaning...
    "%PY_EXE%" scripts\run_data_cleaning.py
    if errorlevel 1 (
        echo ERROR: Data cleaning failed!
        pause
        exit /b 1
    )
)

:: Check and create tokenizer if needed
if not exist "models\tokenizer.pkl" (
    echo Creating tokenizer...
    "%PY_EXE%" scripts\save_tokenizer.py
    if errorlevel 1 (
        echo ERROR: Tokenizer creation failed!
        pause
        exit /b 1
    )
)

echo.
echo Starting Streamlit app...
echo Please wait 10-15 seconds for the app to start...
echo Then manually open: http://127.0.0.1:8501
echo.

:: Run Streamlit (bind explicitly to IPv4 to avoid localhost IPv6 issues)
"%PY_EXE%" -m streamlit run app.py --server.port 8501 --server.address 127.0.0.1

echo.
echo Application closed.
pause
