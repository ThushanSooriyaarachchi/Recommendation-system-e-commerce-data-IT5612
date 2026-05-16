"""KPI card helpers used across pages."""

from __future__ import annotations

import streamlit as st


def _fmt_value(value: float | int | str, kind: str) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "—"
    if kind == "money":
        if abs(value) >= 1_000_000:
            return f"${value/1_000_000:,.2f}M"
        if abs(value) >= 1_000:
            return f"${value/1_000:,.1f}K"
        return f"${value:,.0f}"
    if kind == "int":
        return f"{int(value):,}"
    if kind == "pct":
        return f"{value*100:,.1f}%"
    if kind == "float":
        return f"{value:,.2f}"
    return str(value)


def kpi_card(label: str, value, kind: str = "money", delta: str | None = None, delta_negative: bool = False) -> None:
    st.markdown(
        f"""
        <div class='kpi-card'>
            <div class='kpi-label'>{label}</div>
            <div class='kpi-value'>{_fmt_value(value, kind)}</div>
            {f"<div class='kpi-delta {'negative' if delta_negative else ''}'>{delta}</div>" if delta else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple[str, object, str]]) -> None:
    """Render a row of KPI cards.

    items: list of (label, value, kind) where kind ∈ {money,int,pct,float,raw}.
    """
    cols = st.columns(len(items))
    for col, (label, value, kind) in zip(cols, items):
        with col:
            kpi_card(label, value, kind=kind)
