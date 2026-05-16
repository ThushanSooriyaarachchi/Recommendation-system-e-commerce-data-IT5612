"""Reusable Plotly chart builders."""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PALETTE = px.colors.qualitative.Set2


def line_trend(df: pd.DataFrame, x: str, y: str, title: str, color: str | None = None) -> go.Figure:
    fig = px.line(df, x=x, y=y, color=color, title=title, color_discrete_sequence=PALETTE)
    fig.update_layout(
        margin=dict(l=10, r=10, t=50, b=10),
        plot_bgcolor="white",
        hovermode="x unified",
        height=380,
    )
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=True, gridcolor="#e2e8f0")
    return fig


def bar_top_n(df: pd.DataFrame, label: str, value: str, title: str, n: int = 15, orient: str = "h") -> go.Figure:
    top = df.sort_values(value, ascending=False).head(n)
    if orient == "h":
        fig = px.bar(top.sort_values(value), x=value, y=label, orientation="h", title=title,
                     color_discrete_sequence=[PALETTE[0]])
    else:
        fig = px.bar(top, x=label, y=value, title=title, color_discrete_sequence=[PALETTE[0]])
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), plot_bgcolor="white", height=420)
    return fig


def heatmap(df: pd.DataFrame, x: str, y: str, value: str, title: str) -> go.Figure:
    pivot = df.pivot_table(index=y, columns=x, values=value, aggfunc="sum").fillna(0)
    fig = px.imshow(pivot, aspect="auto", color_continuous_scale="Blues", title=title)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=420)
    return fig


def donut(df: pd.DataFrame, label: str, value: str, title: str) -> go.Figure:
    fig = px.pie(df, names=label, values=value, hole=0.55, title=title,
                 color_discrete_sequence=PALETTE)
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=380)
    return fig


def scatter_rfm(rfm: pd.DataFrame) -> go.Figure:
    fig = px.scatter(
        rfm, x="recency", y="monetary", color="segment",
        size=rfm["frequency"].clip(upper=rfm["frequency"].quantile(0.99)),
        hover_data=["customer_id", "frequency"],
        title="RFM landscape",
        color_discrete_sequence=PALETTE,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=50, b=10), height=460, plot_bgcolor="white")
    return fig
