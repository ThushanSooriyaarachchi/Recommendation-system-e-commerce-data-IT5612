"""Page 6 - Marketing channels & coupon lift."""

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

st.set_page_config(page_title="Marketing & Channels", page_icon="📣", layout="wide")
_, df, _ = bootstrap()
st.title("Marketing & Channels")


channel = (
    df.groupby(["campaign_source", "device_type"], as_index=False, observed=True)
    .agg(revenue=("total_price_usd", "sum"),
         orders=("order_id", "nunique"),
         avg_session=("session_duration_minutes", "mean"))
    .sort_values("revenue", ascending=False)
)
fig = px.bar(channel, x="campaign_source", y="revenue", color="device_type",
             barmode="group", title="Revenue by campaign source x device",
             color_discrete_sequence=px.colors.qualitative.Set2)
fig.update_layout(height=420, plot_bgcolor="white",
                  margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, width="stretch")


traffic = (
    df.groupby("traffic_source", as_index=False, observed=True)
    .agg(revenue=("total_price_usd", "sum"),
         orders=("order_id", "nunique"),
         avg_pages=("pages_visited", "mean"),
         avg_session=("session_duration_minutes", "mean"))
    .sort_values("revenue", ascending=False)
)
c1, c2 = st.columns(2)
with c1:
    fig = px.pie(traffic, names="traffic_source", values="revenue", hole=0.55,
                 title="Revenue share by traffic source",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")
with c2:
    fig = px.scatter(traffic, x="avg_session", y="avg_pages", size="revenue",
                     color="traffic_source",
                     title="Engagement: pages vs session time",
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=420, plot_bgcolor="white",
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")


st.subheader("Coupon lift")
if "coupon_used_bool" in df.columns:
    coup = (
        df.groupby("coupon_used_bool", as_index=False)
        .agg(revenue=("total_price_usd", "sum"),
             orders=("order_id", "nunique"),
             aov=("total_price_usd", "mean"),
             avg_margin=("profit_margin_percent", "mean"))
    )
    coup["coupon"] = coup["coupon_used_bool"].map({True: "With coupon", False: "No coupon"})
    cols = st.columns(3)
    for col, (metric, title) in zip(
        cols,
        [("aov", "AOV"), ("orders", "Orders"), ("avg_margin", "Avg margin %")],
    ):
        with col:
            fig = px.bar(coup, x="coupon", y=metric, color="coupon",
                         title=title, color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=320, showlegend=False,
                              margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white")
            st.plotly_chart(fig, width="stretch")
