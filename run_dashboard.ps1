Set-Location -LiteralPath $PSScriptRoot
Write-Host "Starting UAC forecasting dashboard..."
Write-Host "Keep this window open while using http://127.0.0.1:8501"
Write-Host "Do not press Ctrl+C unless you want to stop the dashboard."
& ".\.venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.fileWatcherType none --browser.gatherUsageStats false --server.enableCORS false --server.enableXsrfProtection false
Write-Host ""
Write-Host "Dashboard stopped with exit code $LASTEXITCODE."
Read-Host "Press Enter to close"
