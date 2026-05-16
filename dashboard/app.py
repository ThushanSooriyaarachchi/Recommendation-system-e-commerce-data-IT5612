"""E-commerce analytics dashboard - Home page.

Run with:
    streamlit run dashboard/app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Make `src` importable when Streamlit launches from the project root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import streamlit as st

from dashboard.components.bootstrap import bootstrap
from dashboard.components.kpis import kpi_row


st.set_page_config(
    page_title="E-Commerce Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

df_full, df, filters = bootstrap()


# ---------------------------------------------------------------------------
# Hero
# ---------------------------------------------------------------------------
st.title("📊 E-Commerce Analytics & ML Dashboard")
st.caption(
    "Interactive multi-page dashboard built on a 1M+ row e-commerce dataset. "
    "Use the sidebar filters to slice by date, country, category and customer segment."
)

# ---------------------------------------------------------------------------
# Headline KPIs
# ---------------------------------------------------------------------------
revenue = float(df["total_price_usd"].sum())
profit = float(df["profit_usd"].sum())
orders = int(df["order_id"].nunique())
customers = int(df["customer_id"].nunique())
aov = revenue / orders if orders else 0
return_rate = float(df["is_returned"].mean()) if "is_returned" in df else 0.0
margin = (profit / revenue) if revenue else 0.0

kpi_row([
    ("Revenue", revenue, "money"),
    ("Profit", profit, "money"),
    ("Orders", orders, "int"),
    ("Customers", customers, "int"),
    ("AOV", aov, "money"),
    ("Return rate", return_rate, "pct"),
    ("Margin", margin, "pct"),
])

st.markdown("<div class='section-title'>Page guide</div>", unsafe_allow_html=True)

PAGES = [
    ("Overview KPIs", "Detailed KPI breakdown and MoM growth"),
    ("Sales Trends", "Time-series, hour×weekday heatmap"),
    ("Geography", "Revenue and orders by country / city"),
    ("Products and Brands", "Top categories, sub-categories, brands"),
    ("Customers RFM", "Recency / Frequency / Monetary segmentation"),
    ("Marketing and Channels", "Coupon lift, traffic source, device"),
    ("Operations", "Shipping, delivery SLA, warehouses"),
    ("Reviews and Sentiment", "Ratings vs sentiment, wordcloud"),
    ("Returns and Fraud", "Return reasons, fraud-risk distribution"),
    ("Forecasting", "Daily revenue forecast (Holt-Winters)"),
    ("Predict Live", "Score new orders against trained models"),
    ("Recommendations", "ALS + content-based + hybrid recommender (Part B)"),
]
for title, desc in PAGES:
    st.markdown(f"- **{title}** — {desc}")

st.divider()

# ---------------------------------------------------------------------------
# Quick preview
# ---------------------------------------------------------------------------
st.markdown("<div class='section-title'>Sample of filtered data</div>", unsafe_allow_html=True)
st.dataframe(
    df.head(50)[
        [
            "order_id", "order_date", "customer_id", "country", "category",
            "brand", "total_price_usd", "profit_usd", "order_status", "rating",
        ]
    ],
    width="stretch",
    height=320,
)

st.caption(
    f"Filtered rows: **{len(df):,}** of **{len(df_full):,}** "
    f"({len(df)/max(len(df_full),1)*100:.1f}%)."
)
