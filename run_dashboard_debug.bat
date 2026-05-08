@echo off
cd /d "%~dp0"
echo Starting UAC forecasting dashboard in debug mode...
echo Keep this window open while using http://127.0.0.1:8501
echo Logs are written to streamlit_debug.log
echo.
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.fileWatcherType none --browser.gatherUsageStats false --server.enableCORS false --server.enableXsrfProtection false --logger.level debug > streamlit_debug.log 2>&1
echo.
echo Dashboard stopped with exit code %ERRORLEVEL%.
echo Last log lines:
powershell -NoProfile -Command "Get-Content -Path streamlit_debug.log -Tail 80"
pause
