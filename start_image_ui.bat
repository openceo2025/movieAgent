@echo off
REM Ensure the script runs from the project root
cd /d %~dp0

REM Activate virtual environment if not already active
if not defined VIRTUAL_ENV (
    if exist .venv\Scripts\activate.bat (
        call .venv\Scripts\activate.bat
    ) else (
        python -m venv .venv
        call .venv\Scripts\activate.bat
        pip install -r requirements.txt
    )
)

REM Set PYTHONPATH to project root
set PYTHONPATH=%~dp0

REM Launch Streamlit app with debug mode
streamlit run movie_agent/image_ui.py -- --debug
