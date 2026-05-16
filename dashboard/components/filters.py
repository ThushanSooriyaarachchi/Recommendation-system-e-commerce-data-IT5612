"""Global sidebar filters shared across pages."""

from __future__ import annotations

import datetime as _dt
from dataclasses import dataclass
from typing import Iterable

import pandas as pd
import streamlit as st


@dataclass
class Filters:
    date_from: _dt.date
    date_to: _dt.date
    countries: list[str]
    categories: list[str]
    segments: list[str]
    sample_size: int | None

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        out = df
        if "order_date" in out.columns:
            od = pd.to_datetime(out["order_date"])
            mask = (od.dt.date >= self.date_from) & (od.dt.date <= self.date_to)
            out = out[mask]
        if self.countries and "country" in out.columns:
            out = out[out["country"].astype("string").isin(self.countries)]
        if self.categories and "category" in out.columns:
            out = out[out["category"].astype("string").isin(self.categories)]
        if self.segments and "customer_segment" in out.columns:
            out = out[out["customer_segment"].astype("string").isin(self.segments)]
        if self.sample_size and len(out) > self.sample_size:
            out = out.sample(n=self.sample_size, random_state=42)
        return out


def _opts(series: pd.Series) -> list[str]:
    return sorted([str(x) for x in series.dropna().unique().tolist()])


def render_sidebar(df: pd.DataFrame) -> Filters:
    st.sidebar.header("Filters")

    od = pd.to_datetime(df["order_date"])
    dmin, dmax = od.min().date(), od.max().date()
    date_range = st.sidebar.date_input(
        "Order date range",
        value=(dmin, dmax),
        min_value=dmin,
        max_value=dmax,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        dfrom, dto = date_range
    else:
        dfrom, dto = dmin, dmax

    countries = st.sidebar.multiselect("Country", _opts(df.get("country", pd.Series(dtype=str))))
    categories = st.sidebar.multiselect("Category", _opts(df.get("category", pd.Series(dtype=str))))
    segments = st.sidebar.multiselect("Customer segment", _opts(df.get("customer_segment", pd.Series(dtype=str))))

    st.sidebar.divider()
    st.sidebar.caption("Performance")
    sample_choice = st.sidebar.radio(
        "Dataset size",
        options=["100k sample", "250k sample", "Full dataset"],
        index=0,
        help="Smaller samples make the dashboard snappier.",
    )
    sample_map = {"100k sample": 100_000, "250k sample": 250_000, "Full dataset": None}
    sample_size = sample_map[sample_choice]

    return Filters(
        date_from=dfrom,
        date_to=dto,
        countries=countries,
        categories=categories,
        segments=segments,
        sample_size=sample_size,
    )
