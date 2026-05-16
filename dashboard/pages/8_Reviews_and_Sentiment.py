"""Page 8 - Reviews & Sentiment."""

from __future__ import annotations

import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import plotly.express as px
import streamlit as st

from dashboard.components.bootstrap import bootstrap

st.set_page_config(page_title="Reviews & Sentiment", page_icon="⭐", layout="wide")
_, df, _ = bootstrap()
st.title("Reviews & Sentiment")


c1, c2 = st.columns([1, 1])
with c1:
    rating = df["rating"].astype("Int8").value_counts().sort_index().reset_index()
    rating.columns = ["rating", "count"]
    fig = px.bar(rating, x="rating", y="count", title="Rating distribution",
                 color="rating", color_continuous_scale="Blues")
    fig.update_layout(height=400, showlegend=False, plot_bgcolor="white",
                      margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")

with c2:
    sent = df["review_sentiment"].astype("string").value_counts(dropna=False).reset_index()
    sent.columns = ["sentiment", "count"]
    fig = px.pie(sent, names="sentiment", values="count", hole=0.55,
                 title="Sentiment mix",
                 color_discrete_sequence=px.colors.qualitative.Set2)
    fig.update_layout(height=400, margin=dict(l=10, r=10, t=50, b=10))
    st.plotly_chart(fig, width="stretch")


st.subheader("Sentiment vs rating")
sr = (
    df.groupby(["rating", "review_sentiment"], as_index=False, observed=True)
    .size().rename(columns={"size": "count"})
)
fig = px.bar(sr, x="rating", y="count", color="review_sentiment", barmode="stack",
             title="Sentiment label distribution by star rating",
             color_discrete_sequence=px.colors.qualitative.Set2)
fig.update_layout(height=420, plot_bgcolor="white",
                  margin=dict(l=10, r=10, t=50, b=10))
st.plotly_chart(fig, width="stretch")


st.subheader("Customer feedback wordcloud")
try:
    from wordcloud import WordCloud
    feedback = (
        df["customer_feedback"].dropna().astype(str)
        .sample(min(20_000, len(df)), random_state=0)
    )
    text = " ".join(feedback.tolist())
    if text.strip():
        wc = WordCloud(width=1400, height=500, background_color="white",
                       colormap="Blues", max_words=120).generate(text)
        buf = BytesIO()
        wc.to_image().save(buf, format="PNG")
        st.image(buf.getvalue(), width="stretch")
    else:
        st.info("No feedback text available for the current filter.")
except ImportError:
    st.info("Install `wordcloud` to enable this visualization.")
