# Predictive Forecasting of Care Load and Placement Demand

This project converts the HHS UAC daily reporting dataset into a forecasting workflow for care load, discharge demand, and early capacity stress indicators.

## Contents

- `app.py` - Streamlit dashboard with model selection, forecast horizon controls, confidence intervals, scenario adjustment, and KPI panels.
- `src/forecasting.py` - Data preparation, feature engineering, forecasting models, evaluation, and KPI calculations.
- `data/HHS_Unaccompanied_Alien_Children_Program.csv` - Local copy of the provided dataset.
- `reports/research_paper.md` - Research paper style write-up with methodology, EDA, model design, and recommendations.
- `reports/executive_summary.md` - Short stakeholder summary for government decision-makers.
- `requirements.txt` - Python dependencies.

## Run

```powershell
.\run_dashboard.ps1
```

Or, without the helper:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501 --server.fileWatcherType none
```

Keep the terminal window open while the dashboard is running. Press `Ctrl+C` only when you want to stop the server. If `localhost` does not resolve correctly on Windows, use `http://127.0.0.1:8501`.

## Analytical Approach

The dashboard uses a strict time-series workflow:

1. Parse and sort dates.
2. Remove blank rows and convert comma-formatted counts to numeric values.
3. Reindex to a continuous daily series.
4. Fill missing days with interpolation or forward-fill based on the dashboard setting.
5. Engineer lag, rolling, flow, and calendar features.
6. Compare baseline, statistical, and machine-learning models using a time-based holdout.
7. Forecast HHS care load and discharge demand with empirical confidence intervals.

## Models

- Naive persistence
- 7-day moving average
- Exponential smoothing
- SARIMA
- Random forest regressor
- Gradient boosting regressor

## KPIs

- Forecast Accuracy %
- Surge Lead Time
- Capacity Breach Probability
- Forecast Stability Index
- 7-day Net Pressure
- 7-day Discharge Demand
