"""Page 7 - Operations: shipping, delivery, warehouses."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.bootstrap import bootstrap

st.set_page_config(page_title="Operations", page_icon="🚚", layout="wide")
_, df, _ = bootstrap()
st.title("Operations")


ship = (
    df.groupby("shipping_method", as_index=False, observed=True)
    .agg(orders=("order_id", "nunique"),
         avg_days=("delivery_days", "mean"),
         avg_cost=("shipping_cost_usd", "mean"),
         on_time=("delivery_status",
                  lambda s: (s.astype("string") == "Delivered").mean()))
    .sort_values("orders", ascending=False)
)

c1, c2 = st.columns(2)
with c1:
    fig = px.bar(ship, x="shipping_method", y="avg_days", color="shipping_method",
                 title="Average delivery days by method",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=400, showlegend=False, plot_bgcolor="white",
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")
with c2:
    fig = px.bar(ship, x="shipping_method", y="on_time", color="shipping_method",
                 title="Delivery success rate",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=400, showlegend=False, plot_bgcolor="white",
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")


st.subheader("Warehouse performance")
wh = (
    df.groupby("warehouse_location", as_index=False, observed=True)
    .agg(orders=("order_id", "nunique"),
         avg_days=("delivery_days", "mean"),
         revenue=("total_price_usd", "sum"))
    .sort_values("orders", ascending=False)
)
fig = px.bar(wh, x="warehouse_location", y="orders", color="avg_days",
             color_continuous_scale="RdYlGn_r",
             title="Orders by warehouse (color = avg delivery days)")
fig.update_layout(height=420, plot_bgcolor="white",
                  margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, width="stretch")


st.subheader("Delivery status mix")
fig = px.pie(df, names="delivery_status", title="Delivery status",
             color_discrete_sequence=px.colors.qualitative.Set2, hole=0.55)
fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, width="stretch")
