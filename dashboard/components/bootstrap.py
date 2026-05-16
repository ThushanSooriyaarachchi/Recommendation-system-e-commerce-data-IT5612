"""Shared per-page bootstrap.

Streamlit multi-page apps let users deep-link to any page directly. If a page
relies on `st.session_state` populated by another page (e.g. Home), it breaks
on first visit. Every page should call :func:`bootstrap` at the top so it can
stand alone.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config
from dashboard.components.filters import Filters, render_sidebar


@st.cache_data(show_spinner="Loading clean dataset…")
def _load_clean_df() -> pd.DataFrame:
    if not config.CLEAN_PARQUET.exists():
        return pd.DataFrame()
    return pd.read_parquet(config.CLEAN_PARQUET)


def _inject_css_once() -> None:
    if st.session_state.get("_css_injected"):
        return
    css_path = ROOT / "dashboard" / "theme" / "style.css"
    if css_path.exists():
        st.markdown(
            f"<style>{css_path.read_text(encoding='utf-8')}</style>",
            unsafe_allow_html=True,
        )
        st.session_state["_css_injected"] = True


def bootstrap(*, require_data: bool = True) -> tuple[pd.DataFrame, pd.DataFrame, Filters | None]:
    """Initialise theme, load the dataset, render the sidebar filters.

    Returns
    -------
    df_full   : pd.DataFrame    -- the full clean dataset (cached)
    df        : pd.DataFrame    -- the dataset after applying sidebar filters
    filters   : Filters | None  -- the parsed sidebar filters (None if no data)
    """
    _inject_css_once()

    df_full = _load_clean_df()

    if df_full.empty:
        if require_data:
            st.error(
                "Clean dataset not found. From a terminal in the project root run "
                "either of:\n\n"
                "```powershell\npython run_pipeline.py        # pandas + sklearn\n"
                "python run_spark_pipeline.py  # PySpark + MLlib + ALS\n```"
            )
            st.stop()
        return df_full, df_full, None

    filters = render_sidebar(df_full)
    df = filters.apply(df_full)

    st.session_state["df_full"] = df_full
    st.session_state["df"] = df
    st.session_state["filters"] = filters
    return df_full, df, filters


__all__ = ["bootstrap"]
