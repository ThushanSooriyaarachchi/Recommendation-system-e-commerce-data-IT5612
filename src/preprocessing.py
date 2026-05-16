"""Cleaning + sanitisation transforms for the e-commerce dataset."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .utils import get_logger, timer

log = get_logger(__name__)

# Sentinel-style placeholder values commonly hiding NaNs.
_NULL_TOKENS = {"", "NA", "N/A", "null", "NULL", "None", "none", "nan"}


def normalize_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Trim whitespace and normalize empty strings to NA across object columns."""
    for col in df.select_dtypes(include=["object", "string"]).columns:
        s = df[col].astype("string").str.strip()
        s = s.where(~s.isin(_NULL_TOKENS), other=pd.NA)
        df[col] = s
    return df


def to_bool(series: pd.Series) -> pd.Series:
    """Convert Yes/No-style series to nullable boolean."""
    mapping = {
        "yes": True, "y": True, "true": True, "1": True,
        "no": False, "n": False, "false": False, "0": False,
    }
    return series.astype("string").str.lower().map(mapping).astype("boolean")


def add_derived_dates(df: pd.DataFrame) -> pd.DataFrame:
    """Robust date features from order_date."""
    if "order_date" not in df.columns:
        return df
    od = pd.to_datetime(df["order_date"], errors="coerce")
    df["order_date"] = od
    df["order_date_only"] = od.dt.date.astype("string")
    df["order_year"] = od.dt.year.astype("Int16")
    df["order_month"] = od.dt.month.astype("Int8")
    df["order_day"] = od.dt.day.astype("Int8")
    df["order_hour"] = od.dt.hour.astype("Int8")
    df["order_dow"] = od.dt.dayofweek.astype("Int8")
    df["order_week"] = od.dt.isocalendar().week.astype("Int16")
    df["order_yearmonth"] = od.dt.to_period("M").astype("string")
    df["is_weekend_bool"] = (df["order_dow"] >= 5).astype("boolean")
    return df


def add_customer_age_in_days(df: pd.DataFrame) -> pd.DataFrame:
    """Tenure of a customer at the time of order."""
    if "account_creation_date" not in df.columns or "order_date" not in df.columns:
        return df
    acd = pd.to_datetime(df["account_creation_date"], errors="coerce")
    od = pd.to_datetime(df["order_date"], errors="coerce")
    tenure = (od - acd).dt.days
    df["customer_tenure_days"] = tenure.clip(lower=0).astype("Int32")
    return df


def fix_numeric_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Cap extreme values that would otherwise distort plots."""
    caps = {
        "session_duration_minutes": (0, 300),
        "delivery_days": (0, 60),
        "age": (0, 100),
        "discount_percent": (0, 90),
        "fraud_risk_score": (0, 100),
        "profit_margin_percent": (-100, 100),
    }
    for col, (lo, hi) in caps.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def add_target_flags(df: pd.DataFrame) -> pd.DataFrame:
    """ML-friendly binary targets derived from raw fields."""
    if "order_status" in df.columns:
        df["is_returned"] = (df["order_status"].astype("string") == "Returned").astype("Int8")
    if "fraud_risk_score" in df.columns:
        df["is_high_fraud"] = (df["fraud_risk_score"] >= 75).astype("Int8")
    if "support_ticket_created" in df.columns:
        df["has_support_ticket"] = to_bool(df["support_ticket_created"]).astype("Int8")
    if "abandoned_cart_before" in df.columns:
        df["had_abandoned_cart"] = to_bool(df["abandoned_cart_before"]).astype("Int8")
    if "coupon_used" in df.columns:
        df["coupon_used_bool"] = to_bool(df["coupon_used"]).astype("Int8")
    return df


def fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Impute the small number of missing fields where it is safe to do so."""
    fill_zero = ["discount_percent", "discount_amount_usd", "tax_usd"]
    for col in fill_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    if "return_reason" in df.columns:
        df["return_reason"] = df["return_reason"].fillna("Not Returned")
    if "customer_feedback" in df.columns:
        df["customer_feedback"] = df["customer_feedback"].fillna("")
    if "coupon_code" in df.columns:
        df["coupon_code"] = df["coupon_code"].fillna("")
    return df


def drop_invalid(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows that cannot be salvaged."""
    before = len(df)
    df = df.dropna(subset=["order_id", "order_date", "customer_id", "total_price_usd"])
    df = df[df["total_price_usd"] >= 0]
    after = len(df)
    if after < before:
        log.info("Dropped %s invalid rows (%.2f%%)", f"{before - after:,}", 100 * (before - after) / before)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """End-to-end cleaning pipeline."""
    with timer("preprocessing.clean"):
        df = normalize_strings(df)
        df = add_derived_dates(df)
        df = add_customer_age_in_days(df)
        df = fix_numeric_outliers(df)
        df = fill_missing(df)
        df = add_target_flags(df)
        df = drop_invalid(df)
        df = df.reset_index(drop=True)
    return df


__all__ = [
    "clean",
    "normalize_strings",
    "to_bool",
    "add_derived_dates",
    "add_customer_age_in_days",
    "fix_numeric_outliers",
    "add_target_flags",
    "fill_missing",
    "drop_invalid",
]
