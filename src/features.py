"""Feature engineering: RFM, aggregates, encodings, model-ready frames."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd

from .utils import get_logger, timer

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# KPI aggregations powering the dashboard
# ---------------------------------------------------------------------------
def kpi_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Daily revenue / profit / orders for the time-series page."""
    g = (
        df.assign(date=pd.to_datetime(df["order_date"]).dt.floor("D"))
        .groupby("date", as_index=False)
        .agg(
            revenue=("total_price_usd", "sum"),
            profit=("profit_usd", "sum"),
            orders=("order_id", "nunique"),
            customers=("customer_id", "nunique"),
            avg_basket=("total_price_usd", "mean"),
            return_rate=("is_returned", "mean"),
        )
        .sort_values("date")
    )
    return g


def kpi_country(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("country", observed=True, as_index=False)
        .agg(
            revenue=("total_price_usd", "sum"),
            profit=("profit_usd", "sum"),
            orders=("order_id", "nunique"),
            customers=("customer_id", "nunique"),
            return_rate=("is_returned", "mean"),
        )
        .sort_values("revenue", ascending=False)
    )


def kpi_category(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(["category", "sub_category"], observed=True, as_index=False)
        .agg(
            revenue=("total_price_usd", "sum"),
            profit=("profit_usd", "sum"),
            orders=("order_id", "nunique"),
            avg_margin=("profit_margin_percent", "mean"),
            avg_rating=("rating", "mean"),
        )
        .sort_values("revenue", ascending=False)
    )


def kpi_brand(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("brand", observed=True, as_index=False)
        .agg(
            revenue=("total_price_usd", "sum"),
            profit=("profit_usd", "sum"),
            orders=("order_id", "nunique"),
            avg_rating=("rating", "mean"),
        )
        .sort_values("revenue", ascending=False)
    )


def kpi_channel(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby(
            ["campaign_source", "traffic_source", "device_type"],
            observed=True,
            as_index=False,
        )
        .agg(
            revenue=("total_price_usd", "sum"),
            orders=("order_id", "nunique"),
            avg_session_min=("session_duration_minutes", "mean"),
            coupon_share=("coupon_used_bool", "mean"),
        )
        .sort_values("revenue", ascending=False)
    )


# ---------------------------------------------------------------------------
# RFM segmentation
# ---------------------------------------------------------------------------
def compute_rfm(df: pd.DataFrame, snapshot: pd.Timestamp | None = None) -> pd.DataFrame:
    """Recency / Frequency / Monetary table per customer."""
    od = pd.to_datetime(df["order_date"])
    snapshot = snapshot or (od.max() + pd.Timedelta(days=1))
    rfm = (
        df.assign(_od=od)
        .groupby("customer_id", as_index=False)
        .agg(
            recency=("_od", lambda s: (snapshot - s.max()).days),
            frequency=("order_id", "nunique"),
            monetary=("total_price_usd", "sum"),
        )
    )

    rfm["R"] = pd.qcut(rfm["recency"], 5, labels=[5, 4, 3, 2, 1], duplicates="drop").astype(int)
    rfm["F"] = pd.qcut(rfm["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    rfm["M"] = pd.qcut(rfm["monetary"], 5, labels=[1, 2, 3, 4, 5], duplicates="drop").astype(int)
    rfm["RFM_score"] = rfm["R"] * 100 + rfm["F"] * 10 + rfm["M"]

    def label(row: pd.Series) -> str:
        r, f, m = row["R"], row["F"], row["M"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        if r >= 4 and f >= 3:
            return "Loyal"
        if r >= 4 and m >= 4:
            return "Big Spenders"
        if r >= 3 and f <= 2:
            return "Promising"
        if r <= 2 and f >= 4:
            return "At Risk"
        if r <= 2 and f <= 2 and m <= 2:
            return "Hibernating"
        return "Needs Attention"

    rfm["segment"] = rfm.apply(label, axis=1)
    return rfm


# ---------------------------------------------------------------------------
# Modelling features
# ---------------------------------------------------------------------------
NUMERIC_FEATURES: list[str] = [
    "age",
    "customer_loyalty_score",
    "total_orders_by_customer",
    "customer_tenure_days",
    "product_rating_avg",
    "product_reviews_count",
    "stock_quantity",
    "unit_price_usd",
    "quantity",
    "discount_percent",
    "discount_amount_usd",
    "total_price_usd",
    "cost_usd",
    "tax_usd",
    "shipping_cost_usd",
    "delivery_days",
    "session_duration_minutes",
    "pages_visited",
    "fraud_risk_score",
    "profit_margin_percent",
    "order_hour",
    "order_dow",
]

CATEGORICAL_FEATURES: list[str] = [
    "gender",
    "customer_segment",
    "country",
    "category",
    "sub_category",
    "brand",
    "payment_method",
    "shipping_method",
    "device_type",
    "traffic_source",
    "campaign_source",
    "order_priority",
]


def build_model_frame(df: pd.DataFrame, target: str, extra_drop: Iterable[str] = ()) -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) ready for a sklearn ColumnTransformer pipeline."""
    cols = [c for c in (NUMERIC_FEATURES + CATEGORICAL_FEATURES) if c in df.columns]
    drop = set(extra_drop) | {target}
    cols = [c for c in cols if c not in drop]
    X = df[cols].copy()
    y = df[target].copy()
    return X, y


def feature_columns(df: pd.DataFrame, target: str) -> tuple[list[str], list[str]]:
    num = [c for c in NUMERIC_FEATURES if c in df.columns and c != target]
    cat = [c for c in CATEGORICAL_FEATURES if c in df.columns and c != target]
    return num, cat


__all__ = [
    "kpi_daily",
    "kpi_country",
    "kpi_category",
    "kpi_brand",
    "kpi_channel",
    "compute_rfm",
    "build_model_frame",
    "feature_columns",
    "NUMERIC_FEATURES",
    "CATEGORICAL_FEATURES",
]
