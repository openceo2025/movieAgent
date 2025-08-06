@echo off
REM Activate movieagent conda environment
call conda activate movieagent

REM Set PYTHONPATH to the project root so movie_agent package resolves
set PYTHONPATH=%~dp0

REM Launch Streamlit app with debug mode
streamlit run movie_agent/image_ui.py -- --debug
