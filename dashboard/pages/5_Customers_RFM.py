"""Page 5 - Customer RFM segmentation."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from src import config
from src.features import compute_rfm
from dashboard.components.bootstrap import bootstrap
from dashboard.components.charts import scatter_rfm

st.set_page_config(page_title="Customers RFM", page_icon="👥", layout="wide")
_, df, _ = bootstrap()
st.title("Customer Segmentation (RFM)")


@st.cache_data(show_spinner="Computing RFM…")
def get_rfm_full() -> pd.DataFrame:
    if config.RFM_PARQUET.exists():
        return pd.read_parquet(config.RFM_PARQUET)
    return pd.DataFrame()


use_filtered = st.toggle(
    "Compute RFM on the filtered slice (slower) — otherwise use the pre-built RFM "
    "table for the full dataset.",
    value=False,
)

if use_filtered:
    rfm = compute_rfm(df)
else:
    rfm = get_rfm_full()
    if rfm.empty:
        rfm = compute_rfm(df)


seg = (
    rfm.groupby("segment", as_index=False)
    .agg(customers=("customer_id", "nunique"),
         avg_recency=("recency", "mean"),
         avg_frequency=("frequency", "mean"),
         avg_monetary=("monetary", "mean"),
         total_revenue=("monetary", "sum"))
    .sort_values("total_revenue", ascending=False)
)

c1, c2 = st.columns([1.1, 1])
with c1:
    fig = px.pie(seg, names="segment", values="customers", hole=0.55,
                 title="Customers per segment",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=440, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")
with c2:
    fig = px.bar(seg.sort_values("total_revenue"),
                 x="total_revenue", y="segment", orientation="h",
                 title="Revenue by segment",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=440, plot_bgcolor="white",
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")


st.subheader("RFM scatter")
sample_n = min(5000, len(rfm))
st.plotly_chart(scatter_rfm(rfm.sample(sample_n, random_state=0)), width="stretch")


with st.expander("Segment summary table"):
    st.dataframe(seg, width="stretch")
with st.expander("Customer-level RFM (first 200 rows)"):
    st.dataframe(rfm.head(200), width="stretch")
