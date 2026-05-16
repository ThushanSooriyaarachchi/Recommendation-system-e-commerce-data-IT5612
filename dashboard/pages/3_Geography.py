"""Page 3 - Geography."""

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

st.set_page_config(page_title="Geography", page_icon="🌍", layout="wide")
_, df, _ = bootstrap()
st.title("Geography")


country = (
    df.groupby("country", as_index=False, observed=True)
    .agg(revenue=("total_price_usd", "sum"),
         profit=("profit_usd", "sum"),
         orders=("order_id", "nunique"),
         customers=("customer_id", "nunique"),
         return_rate=("is_returned", "mean"))
    .sort_values("revenue", ascending=False)
)

c1, c2 = st.columns([2, 1])
with c1:
    fig = px.choropleth(
        country, locations="country", locationmode="country names",
        color="revenue", hover_data=["orders", "customers", "profit"],
        title="Revenue by country", color_continuous_scale="Blues",
    )
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")
with c2:
    fig = px.bar(country.head(15).sort_values("revenue"),
                 x="revenue", y="country", orientation="h",
                 title="Top 15 countries by revenue",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="white")
    st.plotly_chart(fig, width="stretch")

st.subheader("City performance")
city = (
    df.groupby(["country", "city"], as_index=False, observed=True)
    .agg(revenue=("total_price_usd", "sum"),
         orders=("order_id", "nunique"))
    .sort_values("revenue", ascending=False)
    .head(25)
)
fig = px.bar(city.sort_values("revenue"), x="revenue", y="city", color="country",
             orientation="h", title="Top 25 cities",
             color_discrete_sequence=px.colors.qualitative.Set2)
fig.update_layout(height=600, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="white")
st.plotly_chart(fig, width="stretch")

with st.expander("Country table"):
    st.dataframe(country, width="stretch")
