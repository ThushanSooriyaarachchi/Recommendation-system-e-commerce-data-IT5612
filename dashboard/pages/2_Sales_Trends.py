"""Page 2 - Sales trends & seasonality."""

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

st.set_page_config(page_title="Sales Trends", page_icon="📈", layout="wide")
_, df, _ = bootstrap()
st.title("Sales Trends & Seasonality")


gran = st.radio("Granularity", ["Daily", "Weekly", "Monthly"],
                horizontal=True, index=1)
od = pd.to_datetime(df["order_date"])

if gran == "Daily":
    g = od.dt.floor("D")
elif gran == "Weekly":
    g = od.dt.to_period("W").dt.to_timestamp()
else:
    g = od.dt.to_period("M").dt.to_timestamp()

trend = (
    df.assign(_period=g)
    .groupby("_period", as_index=False)
    .agg(revenue=("total_price_usd", "sum"),
         profit=("profit_usd", "sum"),
         orders=("order_id", "nunique"))
    .sort_values("_period")
    .rename(columns={"_period": "period"})
)

fig = px.line(trend, x="period", y=["revenue", "profit"],
              title=f"{gran} revenue and profit",
              color_discrete_sequence=px.colors.qualitative.Set2)
fig.update_layout(height=420, plot_bgcolor="white", hovermode="x unified",
                  margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, width="stretch")

st.subheader("When do customers buy? (orders heatmap)")
hourly = (
    df.assign(hour=od.dt.hour, dow=od.dt.dayofweek)
    .groupby(["dow", "hour"], as_index=False)
    .agg(orders=("order_id", "nunique"))
)
DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
hourly["dow"] = hourly["dow"].map(dict(enumerate(DOW)))
pivot = hourly.pivot(index="dow", columns="hour", values="orders").reindex(DOW).fillna(0)

fig = px.imshow(
    pivot, aspect="auto", color_continuous_scale="Blues",
    labels=dict(x="Hour of day", y="Day of week", color="Orders"),
    title="Orders by hour x weekday",
)
fig.update_layout(height=420, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, width="stretch")

st.subheader("Weekend vs weekday performance")
wk = (
    df.assign(is_weekend=od.dt.dayofweek >= 5)
    .groupby("is_weekend", as_index=False)
    .agg(revenue=("total_price_usd", "sum"),
         orders=("order_id", "nunique"),
         aov=("total_price_usd", "mean"))
)
wk["bucket"] = wk["is_weekend"].map({True: "Weekend", False: "Weekday"})
c1, c2, c3 = st.columns(3)
for col, (metric, title) in zip(
    [c1, c2, c3],
    [("revenue", "Revenue"), ("orders", "Orders"), ("aov", "AOV")],
):
    with col:
        fig = px.bar(wk, x="bucket", y=metric, color="bucket", title=title,
                     color_discrete_sequence=px.colors.qualitative.Set2)
        fig.update_layout(height=320, showlegend=False,
                          margin=dict(l=10, r=10, t=40, b=10), plot_bgcolor="white")
        st.plotly_chart(fig, width="stretch")
