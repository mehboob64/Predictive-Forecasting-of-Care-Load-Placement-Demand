# Predictive Forecasting of Care Load and Placement Demand for the UAC Program

## Abstract

The Unaccompanied Alien Children (UAC) Program requires forward-looking intelligence to manage care capacity under uncertain and rapidly changing conditions. This project applies time-series preparation, feature engineering, statistical forecasting, and machine-learning models to forecast HHS care load and discharge demand. The resulting Streamlit dashboard provides model comparison, confidence intervals, scenario analysis, and early-warning indicators for capacity stress.

## Background

Daily UAC reporting captures apprehensions, CBP custody, transfers into HHS custody, children in HHS care, and discharges from HHS care. These measures describe both stock variables, such as current care load, and flow variables, such as transfers and discharges. The operational challenge is that descriptive reporting alone cannot answer whether tomorrow's or next week's care load will exceed available capacity.

## Problem Statement

The program needs short-term forecasts of children in HHS care, predictive estimates of discharge placement demand, and early warning of capacity stress. Without these capabilities, planners risk delayed shelter expansion, staff burnout, longer lengths of stay, and overcrowding.

## Data Description

The dataset contains daily records with the following fields:

- Date: reporting date.
- Children apprehended and placed in CBP custody: daily intake volume.
- Children in CBP custody: active CBP care load.
- Children transferred out of CBP custody: flow into the HHS system.
- Children in HHS Care: active HHS care load.
- Children discharged from HHS Care: sponsor placement exits.

The source file includes comma-formatted counts and blank trailing rows. Dates are normalized into a continuous daily index. Missing days can be handled through time interpolation or forward-fill masking in the dashboard.

## Methodology

### Time-Series Preparation

The workflow converts the Date field to a datetime index, sorts observations chronologically, removes blank rows, converts count fields to numeric values, and reindexes to a daily calendar. Missing dates are handled through interpolation by default, with forward-fill available for sensitivity testing.

### Feature Engineering

Predictive features include lag values at 1, 7, and 14 days; rolling 7-day and 14-day means and standard deviations; net pressure defined as transfers minus discharges; care-load change; day-of-week; month; and weekend indicators.

### Train-Test Strategy

The workflow uses a strict time-based holdout. Random sampling is not used because it would leak future information into model training. Model performance is reported for 1-7 day, 8-14 day, and full-holdout horizons.

### Forecasting Models

Baseline models include naive persistence and 7-day moving average. Statistical models include exponential smoothing and SARIMA. Machine-learning models include random forest regression and gradient boosting regression trained on lag, rolling, flow, and calendar features.

### Evaluation Metrics

Models are compared using MAE, RMSE, MAPE, and forecast accuracy percentage. Horizon-specific errors help distinguish near-term reliability from medium-term degradation.

## Key Performance Indicators

- Forecast Accuracy %: approximate reliability derived from MAPE.
- Surge Lead Time: days until projected care load reaches the selected warning threshold.
- Capacity Breach Probability: share of forecast days where the 95% upper interval exceeds selected capacity.
- Forecast Stability Index: robustness signal based on forecast volatility.
- 7-day Net Pressure: recent average transfers minus discharges.
- 7-day Discharge Demand: projected placements required during the next week.

## Expected Insights

The most important operational signal is the relationship between transfers and discharges. When transfers out of CBP custody exceed discharges from HHS care for multiple days, the active HHS care load is likely to rise. Conversely, strong discharge throughput can offset new transfers and reduce capacity pressure.

Baseline models are useful benchmarks because they show whether complex models are adding real planning value. Statistical models are appropriate when recent trend and weekly seasonality dominate. Machine-learning models can capture nonlinear relationships between lagged care load, flow pressure, and calendar effects, but they should be monitored for stability.

## Recommendations

1. Refresh forecasts when new daily data are available.
2. Track net pressure as an operational early-warning signal.
3. Use confidence intervals, not only point forecasts, for capacity planning.
4. Compare model errors by horizon before selecting an operational model.
5. Combine forecasts with policy, border activity, facility availability, and sponsor-placement intelligence.
6. Treat scenario analysis as a planning exercise rather than a prediction guarantee.

## Conclusion

This project moves the UAC dataset from historical reporting toward predictive intelligence. The dashboard supports proactive planning by estimating care load, discharge demand, uncertainty, and capacity risk. Used consistently, it can help HHS stakeholders allocate resources earlier, protect child-welfare outcomes, and reduce operational strain.
