@echo off
REM Ensure the script runs from the project root
cd /d %~dp0

REM Activate movieagent conda environment
call conda activate movieagent

REM Launch Streamlit app with debug mode
streamlit run movie_agent/image_ui.py -- --debug
