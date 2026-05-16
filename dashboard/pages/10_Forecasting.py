"""Page 10 - Time-series forecasting (Holt-Winters)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src import config
from src.models import load as load_model
from dashboard.components.bootstrap import bootstrap

st.set_page_config(page_title="Forecasting", page_icon="🔮", layout="wide")
_, df, _ = bootstrap()
st.title("Revenue Forecasting")


horizon = st.slider("Forecast horizon (days)",
                    min_value=14, max_value=120, value=60, step=7)


@st.cache_resource(show_spinner="Loading forecaster…")
def get_forecaster():
    if not config.FORECAST_MODEL_PATH.exists():
        return None
    return load_model(config.FORECAST_MODEL_PATH)


fc = get_forecaster()
if fc is None:
    st.error("Forecaster not trained yet. Run `python run_pipeline.py` first.")
    st.stop()

history = fc.history.copy()
history["date"] = pd.to_datetime(history["date"])

future_index = pd.date_range(start=history["date"].iloc[-1] + pd.Timedelta(days=1),
                             periods=horizon, freq="D")
forecast = fc.fitted.forecast(horizon)
forecast.index = future_index

fig = go.Figure()
fig.add_trace(go.Scatter(x=history["date"], y=history["revenue"],
                         name="History", line=dict(color="#2563eb")))
fig.add_trace(go.Scatter(x=forecast.index, y=forecast.values,
                         name=f"Forecast ({horizon}d)",
                         line=dict(color="#16a34a", dash="dash")))
fig.update_layout(
    title="Daily revenue: history vs forecast",
    height=500, plot_bgcolor="white", hovermode="x unified",
    margin=dict(l=10, r=10, t=50, b=10),
)
st.plotly_chart(fig, width="stretch")


c1, c2, c3 = st.columns(3)
c1.metric("Holdout MAE (14d, training)", f"${fc.metric_mae:,.0f}")
c2.metric("Forecast horizon", f"{horizon} days")
c3.metric("Predicted total revenue", f"${forecast.sum():,.0f}")


with st.expander("Forecast values"):
    st.dataframe(
        pd.DataFrame({"date": forecast.index, "predicted_revenue": forecast.values}),
        width="stretch",
    )
