"""Smoke tests for the preprocessing pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src.preprocessing import clean, to_bool


def test_to_bool_handles_yes_no():
    s = pd.Series(["Yes", "no", "TRUE", "0", None])
    out = to_bool(s)
    assert out.tolist()[:4] == [True, False, True, False]
    assert out.isna().iloc[-1]


def test_clean_drops_invalid_rows():
    df = pd.DataFrame({
        "order_id": ["ORD-1", "ORD-2", "ORD-3", None],
        "order_date": ["2025-01-01", "2025-01-02", "not-a-date", "2025-01-04"],
        "account_creation_date": ["2024-01-01", "2024-06-01", "2024-01-01", "2023-01-01"],
        "customer_id": ["C1", "C2", "C3", "C4"],
        "total_price_usd": [10.0, 20.0, 30.0, 40.0],
        "order_status": ["Completed", "Returned", "Completed", "Completed"],
        "fraud_risk_score": [10, 80, 50, 90],
        "support_ticket_created": ["No", "Yes", "No", "Yes"],
        "abandoned_cart_before": ["No", "Yes", "No", "Yes"],
        "coupon_used": ["No", "Yes", "No", "Yes"],
    })
    out = clean(df)
    assert "is_returned" in out.columns
    assert "is_high_fraud" in out.columns
    # ORD-2 is the Returned one, with a valid date — it should survive cleaning.
    assert out["is_returned"].sum() >= 1
    # Rows with bad date or null id must be dropped.
    assert len(out) == 2
