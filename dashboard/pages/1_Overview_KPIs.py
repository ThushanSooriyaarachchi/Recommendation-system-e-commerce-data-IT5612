"""Page 1 - Overview KPIs with MoM growth."""

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

st.set_page_config(page_title="Overview KPIs", page_icon="📊", layout="wide")
_, df, _ = bootstrap()
st.title("Overview KPIs")


revenue = float(df["total_price_usd"].sum())
profit = float(df["profit_usd"].sum())
orders = int(df["order_id"].nunique())
customers = int(df["customer_id"].nunique())
aov = revenue / orders if orders else 0
return_rate = float(df["is_returned"].mean()) if "is_returned" in df else 0.0
margin = (profit / revenue) if revenue else 0.0
fraud_share = float((df["fraud_risk_score"] >= 75).mean()) if "fraud_risk_score" in df else 0.0

kpi_row([
    ("Revenue", revenue, "money"),
    ("Profit", profit, "money"),
    ("Margin", margin, "pct"),
    ("AOV", aov, "money"),
])
kpi_row([
    ("Orders", orders, "int"),
    ("Customers", customers, "int"),
    ("Return rate", return_rate, "pct"),
    ("High fraud share", fraud_share, "pct"),
])

st.divider()

od = pd.to_datetime(df["order_date"])
monthly = (
    df.assign(month=od.dt.to_period("M").dt.to_timestamp())
    .groupby("month", as_index=False)
    .agg(revenue=("total_price_usd", "sum"), profit=("profit_usd", "sum"),
         orders=("order_id", "nunique"))
    .sort_values("month")
)
monthly["revenue_mom"] = monthly["revenue"].pct_change()

c1, c2 = st.columns([2, 1])
with c1:
    fig = px.bar(monthly, x="month", y=["revenue", "profit"], barmode="group",
                 title="Revenue & profit by month",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="white")
    st.plotly_chart(fig, width="stretch")
with c2:
    fig = px.line(monthly, x="month", y="revenue_mom", markers=True,
                  title="Month-over-month revenue growth")
    fig.update_yaxes(tickformat=".0%")
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="white")
    st.plotly_chart(fig, width="stretch")

st.subheader("Order status mix")
status = df["order_status"].astype("string").value_counts(dropna=False).reset_index()
status.columns = ["order_status", "count"]
fig = px.bar(status, x="order_status", y="count", color="order_status",
             color_discrete_sequence=px.colors.qualitative.Set2)
fig.update_layout(height=320, showlegend=False,
                  margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor="white")
st.plotly_chart(fig, width="stretch")

with st.expander("View monthly KPI table"):
    st.dataframe(monthly, width="stretch")
