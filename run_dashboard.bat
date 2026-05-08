@echo off
cd /d "%~dp0"
echo Starting UAC forecasting dashboard...
echo Keep this window open while using http://127.0.0.1:8501
echo Do not press Ctrl+C unless you want to stop the dashboard.
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.fileWatcherType none --browser.gatherUsageStats false --server.enableCORS false --server.enableXsrfProtection false
echo.
echo Dashboard stopped with exit code %ERRORLEVEL%.
pause
