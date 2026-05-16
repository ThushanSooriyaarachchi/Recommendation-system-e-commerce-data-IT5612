"""Page 4 - Products and Brands."""

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

st.set_page_config(page_title="Products & Brands", page_icon="🛒", layout="wide")
_, df, _ = bootstrap()
st.title("Products & Brands")


cat = (
    df.groupby(["category", "sub_category"], as_index=False, observed=True)
    .agg(revenue=("total_price_usd", "sum"),
         profit=("profit_usd", "sum"),
         orders=("order_id", "nunique"),
         avg_margin=("profit_margin_percent", "mean"),
         avg_rating=("rating", "mean"))
    .sort_values("revenue", ascending=False)
)

st.subheader("Category x sub-category treemap")
fig = px.treemap(
    cat, path=["category", "sub_category"], values="revenue", color="avg_margin",
    color_continuous_scale="RdYlGn",
    color_continuous_midpoint=cat["avg_margin"].mean(),
    title="Revenue treemap (color = avg margin %)",
)
fig.update_layout(height=520, margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, width="stretch")


brand = (
    df.groupby("brand", as_index=False, observed=True)
    .agg(revenue=("total_price_usd", "sum"),
         profit=("profit_usd", "sum"),
         orders=("order_id", "nunique"),
         avg_rating=("rating", "mean"))
    .sort_values("revenue", ascending=False)
)

c1, c2 = st.columns(2)
with c1:
    fig = px.bar(brand.head(15).sort_values("revenue"),
                 x="revenue", y="brand", orientation="h",
                 title="Top 15 brands by revenue",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=480, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="white")
    st.plotly_chart(fig, width="stretch")
with c2:
    fig = px.scatter(brand, x="revenue", y="avg_rating", size="orders", color="brand",
                     title="Brand: revenue vs rating", hover_name="brand",
                     color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=480, showlegend=False, plot_bgcolor="white",
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")

st.subheader("Top products")
top_products = (
    df.groupby("product_name", as_index=False, observed=True)
    .agg(revenue=("total_price_usd", "sum"),
         orders=("order_id", "nunique"),
         avg_rating=("rating", "mean"))
    .sort_values("revenue", ascending=False)
    .head(20)
)
st.dataframe(top_products, width="stretch", height=420)
