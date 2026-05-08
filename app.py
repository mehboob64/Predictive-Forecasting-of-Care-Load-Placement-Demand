from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.forecasting import (
    TARGET_CARE,
    TARGET_DISCHARGE,
    capacity_kpis,
    evaluate_models,
    get_model_registry,
    load_uac_data,
    make_forecast,
)


ROOT = Path(__file__).parent
DATA_PATH = ROOT / "data" / "HHS_Unaccompanied_Alien_Children_Program.csv"


st.set_page_config(
    page_title="UAC Care Load Forecasting",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)


@st.cache_data(show_spinner=False)
def cached_load(fill_method: str) -> pd.DataFrame:
    return load_uac_data(DATA_PATH, fill_method)


st.title("Predictive Forecasting of Care Load and Placement Demand")
st.caption("Forward-looking planning dashboard for UAC care load, discharge demand, and capacity stress.")

with st.sidebar:
    st.header("Forecast Controls")
    fill_method = st.radio("Missing-day handling", ["Interpolate missing days", "Forward fill reported values"])
    horizon = st.slider("Forecast horizon", 7, 60, 21, 1)
    test_days = st.slider("Holdout window for evaluation", 14, 120, 45, 1)
    model_names = list(get_model_registry(TARGET_CARE).keys())
    selected_models = st.multiselect(
        "Compare models",
        model_names,
        default=["Naive persistence", "7-day moving average"],
    )
    primary_model = st.selectbox("Primary forecast model", selected_models or model_names, index=0)
    st.header("Scenario")
    capacity = st.number_input("Available HHS care capacity", min_value=1, value=3500, step=100)
    warning_threshold = st.slider("Early-warning threshold", 0.70, 1.00, 0.90, 0.01)
    intake_scenario = st.slider("Scenario pressure adjustment", -30, 50, 0, 5, help="Percent adjustment to projected care load.")

df = cached_load(fill_method)

if not selected_models:
    st.warning("Select at least one model to compare.")
    st.stop()

care_forecast = make_forecast(df, TARGET_CARE, primary_model, horizon, test_days)
discharge_forecast = make_forecast(df, TARGET_DISCHARGE, primary_model, horizon, test_days)

scenario_factor = 1 + intake_scenario / 100
scenario_forecast = care_forecast.forecast.copy()
scenario_forecast[["forecast", "lower_95", "upper_95"]] *= scenario_factor

kpis = capacity_kpis(df, scenario_forecast, discharge_forecast.forecast, capacity, warning_threshold)
metrics = evaluate_models(df, TARGET_CARE, selected_models, test_days)

latest_date = df.index.max().date()
reported_share = df["was_reported"].mean() * 100
latest_care = df[TARGET_CARE].iloc[-1]
latest_discharge = df[TARGET_DISCHARGE].iloc[-1]

metric_cols = st.columns(5)
metric_cols[0].metric("Latest HHS Care Load", f"{latest_care:,.0f}", help=f"Latest normalized date: {latest_date}")
metric_cols[1].metric("Latest Discharges", f"{latest_discharge:,.0f}")
metric_cols[2].metric("Surge Lead Time", f"{kpis['Surge Lead Time']} days")
metric_cols[3].metric("Capacity Breach Probability", f"{kpis['Capacity Breach Probability']:.0%}")
metric_cols[4].metric("Forecast Stability Index", f"{kpis['Forecast Stability Index']:.1f}")

tabs = st.tabs(["Care Load Forecast", "Discharge Demand", "Model Comparison", "Data and Signals"])

with tabs[0]:
    chart_df = df.tail(180)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=chart_df.index, y=chart_df[TARGET_CARE], name="Historical HHS care", mode="lines"))
    fig.add_trace(go.Scatter(x=scenario_forecast.index, y=scenario_forecast["forecast"], name=f"{primary_model} forecast", mode="lines"))
    fig.add_trace(
        go.Scatter(
            x=list(scenario_forecast.index) + list(scenario_forecast.index[::-1]),
            y=list(scenario_forecast["upper_95"]) + list(scenario_forecast["lower_95"][::-1]),
            fill="toself",
            fillcolor="rgba(31, 119, 180, 0.18)",
            line=dict(color="rgba(255,255,255,0)"),
            hoverinfo="skip",
            name="95% interval",
        )
    )
    fig.add_hline(y=capacity, line_dash="dash", line_color="#d62728", annotation_text="Capacity")
    fig.add_hline(y=capacity * warning_threshold, line_dash="dot", line_color="#ff7f0e", annotation_text="Warning threshold")
    fig.update_layout(height=520, yaxis_title="Children in HHS care", hovermode="x unified")
    st.plotly_chart(fig, width="stretch")

with tabs[1]:
    col_a, col_b = st.columns([2, 1])
    with col_a:
        discharge = discharge_forecast.forecast
        fig = go.Figure()
        fig.add_trace(go.Bar(x=discharge.index, y=discharge["forecast"], name="Forecast discharges"))
        fig.add_trace(go.Scatter(x=df.tail(90).index, y=df[TARGET_DISCHARGE].tail(90), name="Historical discharges", mode="lines"))
        fig.update_layout(height=460, yaxis_title="Children discharged", hovermode="x unified")
        st.plotly_chart(fig, width="stretch")
    with col_b:
        st.subheader("Placement Demand")
        st.metric("Next 7 days", f"{discharge['forecast'].head(7).sum():,.0f}")
        st.metric("Full horizon", f"{discharge['forecast'].sum():,.0f}")
        st.metric("7-day net pressure", f"{kpis['7-day Net Pressure']:,.1f}")
        st.write("Positive net pressure indicates transfers into HHS are running above discharge placements.")

with tabs[2]:
    st.subheader("Time-Based Holdout Evaluation")
    display_metrics = metrics.copy()
    for col in ["MAE", "RMSE", "MAPE %", "Forecast Accuracy %"]:
        display_metrics[col] = display_metrics[col].astype(float).round(2)
    st.dataframe(display_metrics[["Model", "Horizon", "MAE", "RMSE", "MAPE %", "Forecast Accuracy %"]], width="stretch")

    full = display_metrics[display_metrics["Horizon"] == "Full holdout"].sort_values("MAE")
    fig = go.Figure()
    fig.add_trace(go.Bar(x=full["Model"], y=full["MAE"], name="MAE"))
    fig.add_trace(go.Bar(x=full["Model"], y=full["RMSE"], name="RMSE"))
    fig.update_layout(barmode="group", height=420, yaxis_title="Error")
    st.plotly_chart(fig, width="stretch")

with tabs[3]:
    col_a, col_b = st.columns([1, 1])
    with col_a:
        st.subheader("Reporting Coverage")
        st.metric("Date range", f"{df.index.min().date()} to {df.index.max().date()}")
        st.metric("Reported-day coverage", f"{reported_share:.1f}%")
        st.metric("Daily rows after normalization", f"{len(df):,}")
        st.dataframe(df.tail(20), width="stretch")
    with col_b:
        st.subheader("Flow Signals")
        signals = df[[TARGET_CARE, TARGET_DISCHARGE, "net_pressure", "care_change"]].tail(120)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=signals.index, y=signals["net_pressure"], name="Transfers minus discharges", mode="lines"))
        fig.add_trace(go.Scatter(x=signals.index, y=signals["care_change"], name="Daily care-load change", mode="lines"))
        fig.add_hline(y=0, line_color="#666", line_width=1)
        fig.update_layout(height=420, hovermode="x unified")
        st.plotly_chart(fig, width="stretch")

st.divider()
st.caption(
    "Models use strict time-based splitting and forecast from historical observations only. "
    "Confidence bands are empirical holdout-residual intervals, intended for planning risk rather than formal guarantees."
)
