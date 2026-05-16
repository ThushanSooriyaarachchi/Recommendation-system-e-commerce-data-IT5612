"""Page 9 - Returns & Fraud."""

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
from dashboard.components.kpis import kpi_row

st.set_page_config(page_title="Returns & Fraud", page_icon="⚠️", layout="wide")
_, df, _ = bootstrap()
st.title("Returns & Fraud")


return_rate = float(df["is_returned"].mean()) if "is_returned" in df else 0.0
returned_revenue = float(df.loc[df.get("is_returned", 0) == 1, "total_price_usd"].sum())
high_fraud_share = float((df["fraud_risk_score"] >= 75).mean()) if "fraud_risk_score" in df else 0.0
avg_fraud = float(df["fraud_risk_score"].mean()) if "fraud_risk_score" in df else 0.0

kpi_row([
    ("Return rate", return_rate, "pct"),
    ("Returned revenue", returned_revenue, "money"),
    ("Avg fraud score", avg_fraud, "float"),
    ("High-risk share", high_fraud_share, "pct"),
])


st.subheader("Return reasons")
ret = df[df["return_reason"].astype("string") != "Not Returned"]
if not ret.empty:
    rc = ret["return_reason"].astype("string").value_counts().reset_index()
    rc.columns = ["reason", "count"]
    rc["cum_share"] = rc["count"].cumsum() / rc["count"].sum()
    fig = px.bar(rc, x="reason", y="count", title="Return reasons (Pareto)",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.add_scatter(x=rc["reason"], y=rc["cum_share"] * rc["count"].max(),
                    mode="lines+markers", name="Cumulative share",
                    line=dict(color="#dc2626", width=2))
    fig.update_layout(height=420, plot_bgcolor="white",
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")
else:
    st.info("No returns in the current filter.")


st.subheader("Fraud risk score distribution")
fig = px.histogram(df, x="fraud_risk_score", nbins=40,
                   title="Distribution of fraud_risk_score",
                   color_discrete_sequence=[px.colors.qualitative.Set2[2]])
fig.update_layout(height=400, plot_bgcolor="white",
                  margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, width="stretch")


rc_cat = (
    df.groupby("category", as_index=False, observed=True)
    .agg(return_rate=("is_returned", "mean"),
         orders=("order_id", "nunique"))
    .sort_values("return_rate", ascending=False)
)
fig = px.bar(rc_cat, x="category", y="return_rate", color="orders",
             color_continuous_scale="Blues",
             title="Return rate by category")
fig.update_yaxes(tickformat=".1%")
fig.update_layout(height=400, plot_bgcolor="white",
                  margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, width="stretch")
