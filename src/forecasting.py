from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.statespace.sarimax import SARIMAX
except Exception:  # pragma: no cover - app can still run without statsmodels
    ExponentialSmoothing = None
    SARIMAX = None


DATE_COL = "Date"
TARGET_CARE = "Children in HHS Care"
TARGET_DISCHARGE = "Children discharged from HHS Care"
TRANSFER_COL = "Children transferred out of CBP custody"
INTAKE_COL = "Children apprehended and placed in CBP custody"
CBP_CARE_COL = "Children in CBP custody"


@dataclass
class ForecastResult:
    name: str
    history: pd.DataFrame
    forecast: pd.DataFrame
    metrics: pd.DataFrame


def load_uac_data(path: str | Path, fill_method: str = "Interpolate missing days") -> pd.DataFrame:
    raw = pd.read_csv(path)
    raw = raw.dropna(how="all")
    raw = raw[raw[DATE_COL].notna()].copy()
    raw[DATE_COL] = pd.to_datetime(raw[DATE_COL], errors="coerce")
    raw = raw.dropna(subset=[DATE_COL])

    for col in raw.columns:
        if col == DATE_COL:
            continue
        raw[col] = (
            raw[col]
            .astype(str)
            .str.replace(",", "", regex=False)
            .str.replace("*", "", regex=False)
            .replace({"": np.nan, "nan": np.nan})
        )
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    raw = raw.sort_values(DATE_COL).drop_duplicates(DATE_COL, keep="last")
    raw = raw.set_index(DATE_COL)
    full_index = pd.date_range(raw.index.min(), raw.index.max(), freq="D")
    daily = raw.reindex(full_index)
    daily.index.name = DATE_COL
    daily["was_reported"] = daily[TARGET_CARE].notna()

    numeric_cols = [col for col in daily.columns if col != "was_reported"]
    if fill_method == "Interpolate missing days":
        daily[numeric_cols] = daily[numeric_cols].interpolate("time").ffill().bfill()
    else:
        daily[numeric_cols] = daily[numeric_cols].ffill().bfill()

    daily["net_pressure"] = daily[TRANSFER_COL] - daily[TARGET_DISCHARGE]
    daily["care_change"] = daily[TARGET_CARE].diff().fillna(0)
    daily["day_of_week"] = daily.index.dayofweek
    daily["month"] = daily.index.month
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6]).astype(int)
    return daily


def add_forecast_features(df: pd.DataFrame, target: str) -> pd.DataFrame:
    out = df.copy()
    for lag in (1, 7, 14):
        out[f"{target}_lag_{lag}"] = out[target].shift(lag)
    for window in (7, 14):
        out[f"{target}_roll_mean_{window}"] = out[target].shift(1).rolling(window).mean()
        out[f"{target}_roll_std_{window}"] = out[target].shift(1).rolling(window).std()
        out[f"net_pressure_roll_mean_{window}"] = out["net_pressure"].shift(1).rolling(window).mean()
    out["transfer_lag_1"] = out[TRANSFER_COL].shift(1)
    out["discharge_lag_1"] = out[TARGET_DISCHARGE].shift(1)
    return out


def train_test_split_time(df: pd.DataFrame, test_days: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    test_days = min(test_days, max(14, len(df) // 3))
    return df.iloc[:-test_days].copy(), df.iloc[-test_days:].copy()


def _metrics(actual: pd.Series, pred: pd.Series, horizon_label: str) -> dict[str, float | str]:
    actual, pred = actual.align(pred, join="inner")
    mae = mean_absolute_error(actual, pred)
    rmse = mean_squared_error(actual, pred) ** 0.5
    denom = actual.replace(0, np.nan).abs()
    mape = ((actual - pred).abs() / denom).mean() * 100
    accuracy = max(0.0, 100.0 - float(mape)) if pd.notna(mape) else np.nan
    return {
        "Horizon": horizon_label,
        "MAE": mae,
        "RMSE": rmse,
        "MAPE %": mape,
        "Forecast Accuracy %": accuracy,
    }


def forecast_naive(train: pd.Series, horizon: int) -> np.ndarray:
    return np.repeat(float(train.iloc[-1]), horizon)


def forecast_moving_average(train: pd.Series, horizon: int, window: int = 7) -> np.ndarray:
    return np.repeat(float(train.tail(window).mean()), horizon)


def forecast_exp_smoothing(train: pd.Series, horizon: int) -> np.ndarray:
    if ExponentialSmoothing is None or len(train) < 28:
        return forecast_moving_average(train, horizon, 14)
    model = ExponentialSmoothing(
        train,
        trend="add",
        seasonal="add",
        seasonal_periods=7,
        initialization_method="estimated",
    )
    fit = model.fit(optimized=True)
    return np.asarray(fit.forecast(horizon))


def forecast_sarima(train: pd.Series, horizon: int) -> np.ndarray:
    if SARIMAX is None or len(train) < 42:
        return forecast_exp_smoothing(train, horizon)
    model = SARIMAX(
        train,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fit = model.fit(disp=False)
    return np.asarray(fit.forecast(horizon))


def _ml_feature_cols(target: str) -> list[str]:
    return [
        f"{target}_lag_1",
        f"{target}_lag_7",
        f"{target}_lag_14",
        f"{target}_roll_mean_7",
        f"{target}_roll_std_7",
        f"{target}_roll_mean_14",
        f"{target}_roll_std_14",
        "net_pressure_roll_mean_7",
        "net_pressure_roll_mean_14",
        "transfer_lag_1",
        "discharge_lag_1",
        "day_of_week",
        "month",
        "is_weekend",
    ]


def forecast_ml(
    df: pd.DataFrame,
    target: str,
    horizon: int,
    estimator: RandomForestRegressor | GradientBoostingRegressor,
) -> np.ndarray:
    featured = add_forecast_features(df, target).dropna()
    feature_cols = _ml_feature_cols(target)
    estimator.fit(featured[feature_cols], featured[target])

    working = df.copy()
    preds: list[float] = []
    for step in range(1, horizon + 1):
        next_date = working.index[-1] + pd.Timedelta(days=1)
        row = working.iloc[-1].copy()
        row.name = next_date
        row["day_of_week"] = next_date.dayofweek
        row["month"] = next_date.month
        row["is_weekend"] = int(next_date.dayofweek in [5, 6])
        row[TRANSFER_COL] = working[TRANSFER_COL].tail(7).mean()
        row[TARGET_DISCHARGE] = working[TARGET_DISCHARGE].tail(7).mean()
        row["net_pressure"] = row[TRANSFER_COL] - row[TARGET_DISCHARGE]
        working = pd.concat([working, row.to_frame().T])
        candidate = add_forecast_features(working, target).iloc[[-1]]
        pred = float(estimator.predict(candidate[feature_cols])[0])
        working.loc[next_date, target] = max(0, pred)
        preds.append(max(0, pred))
    return np.asarray(preds)


def get_model_registry(target: str) -> dict[str, Callable[[pd.DataFrame, int], np.ndarray]]:
    return {
        "Naive persistence": lambda df, h: forecast_naive(df[target], h),
        "7-day moving average": lambda df, h: forecast_moving_average(df[target], h, 7),
        "Exponential smoothing": lambda df, h: forecast_exp_smoothing(df[target], h),
        "SARIMA": lambda df, h: forecast_sarima(df[target], h),
        "Random forest": lambda df, h: forecast_ml(
            df,
            target,
            h,
            RandomForestRegressor(n_estimators=250, min_samples_leaf=3, random_state=42),
        ),
        "Gradient boosting": lambda df, h: forecast_ml(
            df,
            target,
            h,
            GradientBoostingRegressor(random_state=42),
        ),
    }


def evaluate_models(df: pd.DataFrame, target: str, model_names: list[str], test_days: int) -> pd.DataFrame:
    train, test = train_test_split_time(df, test_days)
    registry = get_model_registry(target)
    rows = []
    horizons = {"1-7 days": 7, "8-14 days": 14, "Full holdout": len(test)}

    for name in model_names:
        preds = pd.Series(registry[name](train, len(test)), index=test.index, name=name)
        for label, end in horizons.items():
            end = min(end, len(test))
            if label == "8-14 days":
                segment_actual = test[target].iloc[7:end]
                segment_pred = preds.iloc[7:end]
            else:
                segment_actual = test[target].iloc[:end]
                segment_pred = preds.iloc[:end]
            if len(segment_actual) == 0:
                continue
            row = _metrics(segment_actual, segment_pred, label)
            row["Model"] = name
            rows.append(row)
    return pd.DataFrame(rows)


def make_forecast(
    df: pd.DataFrame,
    target: str,
    model_name: str,
    horizon: int,
    test_days: int,
) -> ForecastResult:
    registry = get_model_registry(target)
    metrics = evaluate_models(df, target, [model_name], test_days)
    future_index = pd.date_range(df.index.max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    pred = registry[model_name](df, horizon)

    train, test = train_test_split_time(df, test_days)
    holdout_pred = pd.Series(registry[model_name](train, len(test)), index=test.index)
    residual_std = float((test[target] - holdout_pred).std())
    if not np.isfinite(residual_std) or residual_std == 0:
        residual_std = float(df[target].diff().dropna().std())
    if not np.isfinite(residual_std) or residual_std == 0:
        residual_std = 1.0

    steps = np.arange(1, horizon + 1)
    interval = 1.96 * residual_std * np.sqrt(steps)
    forecast = pd.DataFrame(
        {
            "forecast": pred,
            "lower_95": np.maximum(0, pred - interval),
            "upper_95": pred + interval,
        },
        index=future_index,
    )
    return ForecastResult(model_name, df[[target]].copy(), forecast, metrics)


def capacity_kpis(
    df: pd.DataFrame,
    care_forecast: pd.DataFrame,
    discharge_forecast: pd.DataFrame,
    capacity: int,
    warning_threshold: float,
) -> dict[str, float | int | str]:
    projected = care_forecast["forecast"]
    upper = care_forecast["upper_95"]
    breach_days = int((upper > capacity).sum())
    breach_probability = breach_days / max(1, len(upper))
    above_threshold = projected[projected >= capacity * warning_threshold]
    lead_time = int((above_threshold.index[0] - df.index.max()).days) if not above_threshold.empty else 0
    stability = float(100 / (1 + projected.pct_change().dropna().std() * 100))
    net_pressure_now = float((df[TRANSFER_COL] - df[TARGET_DISCHARGE]).tail(7).mean())
    discharge_demand = float(discharge_forecast["forecast"].head(7).sum())
    return {
        "Capacity Breach Probability": breach_probability,
        "Surge Lead Time": lead_time,
        "Forecast Stability Index": stability,
        "7-day Net Pressure": net_pressure_now,
        "7-day Discharge Demand": discharge_demand,
    }
