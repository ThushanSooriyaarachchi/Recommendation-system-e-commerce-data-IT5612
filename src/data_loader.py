"""Load the raw CSV efficiently and produce a typed clean parquet."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from . import config
from .utils import get_logger, timer

log = get_logger(__name__)

# Explicit dtypes for memory efficiency on the 1M+ row file.
DTYPES: dict[str, str] = {
    "order_id": "string",
    "order_year": "Int16",
    "order_month": "Int8",
    "order_day": "Int8",
    "order_hour": "Int8",
    "order_minute": "Int8",
    "order_second": "Int8",
    "is_weekend": "string",
    "order_status": "category",
    "return_reason": "string",
    "customer_id": "string",
    "customer_name": "string",
    "gender": "category",
    "age": "Int16",
    "customer_segment": "category",
    "country": "category",
    "city": "string",
    "customer_loyalty_score": "float32",
    "total_orders_by_customer": "Int32",
    "product_id": "string",
    "product_name": "string",
    "category": "category",
    "sub_category": "category",
    "brand": "category",
    "product_rating_avg": "float32",
    "product_reviews_count": "Int32",
    "stock_quantity": "Int32",
    "unit_price_usd": "float32",
    "quantity": "Int16",
    "discount_percent": "float32",
    "discount_amount_usd": "float32",
    "total_price_usd": "float32",
    "cost_usd": "float32",
    "profit_usd": "float32",
    "tax_usd": "float32",
    "currency": "category",
    "payment_method": "category",
    "payment_status": "category",
    "installment_plan": "category",
    "shipping_method": "category",
    "shipping_cost_usd": "float32",
    "delivery_days": "Int16",
    "shipping_country": "category",
    "warehouse_location": "category",
    "delivery_status": "category",
    "rating": "Int8",
    "review_sentiment": "category",
    "customer_feedback": "string",
    "coupon_used": "category",
    "coupon_code": "string",
    "campaign_source": "category",
    "device_type": "category",
    "traffic_source": "category",
    "session_duration_minutes": "float32",
    "pages_visited": "Int16",
    "abandoned_cart_before": "category",
    "fraud_risk_score": "float32",
    "profit_margin_percent": "float32",
    "order_priority": "category",
    "support_ticket_created": "category",
}

PARSE_DATES: list[str] = ["order_date", "account_creation_date"]


def iter_chunks(csv_path: Path | None = None, chunksize: int | None = None) -> Iterator[pd.DataFrame]:
    """Yield raw CSV chunks with sane dtypes."""
    csv_path = csv_path or config.RAW_CSV
    chunksize = chunksize or config.CHUNK_SIZE
    if not csv_path.exists():
        raise FileNotFoundError(f"Raw CSV not found: {csv_path}")
    log.info("Reading %s in %d-row chunks", csv_path.name, chunksize)
    yield from pd.read_csv(
        csv_path,
        dtype=DTYPES,
        parse_dates=PARSE_DATES,
        chunksize=chunksize,
        low_memory=False,
    )


def load_raw(csv_path: Path | None = None) -> pd.DataFrame:
    """Load the full CSV into one DataFrame (concatenating chunks)."""
    with timer("load_raw csv"):
        frames = list(iter_chunks(csv_path=csv_path))
        df = pd.concat(frames, ignore_index=True)
    log.info("Loaded %s rows x %s cols", f"{len(df):,}", df.shape[1])
    return df


def load_clean() -> pd.DataFrame:
    """Read the cleaned parquet (must be built first via run_pipeline.py)."""
    if not config.CLEAN_PARQUET.exists():
        raise FileNotFoundError(
            f"{config.CLEAN_PARQUET} not found. Run `python run_pipeline.py` first."
        )
    return pd.read_parquet(config.CLEAN_PARQUET)


def memory_mb(df: pd.DataFrame) -> float:
    return float(df.memory_usage(deep=True).sum()) / 1024**2


def downcast_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce float64/int64 to the smallest safe type."""
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="float")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df


__all__ = [
    "iter_chunks",
    "load_raw",
    "load_clean",
    "memory_mb",
    "downcast_numeric",
    "DTYPES",
    "PARSE_DATES",
]
