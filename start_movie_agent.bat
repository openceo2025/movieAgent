@echo off
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

REM Set PYTHONPATH to the project root so movie_agent package resolves
set PYTHONPATH=%~dp0

REM Launch Streamlit app with debug mode
streamlit run movie_agent/app.py -- --debug
