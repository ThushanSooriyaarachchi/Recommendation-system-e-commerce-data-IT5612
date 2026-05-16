"""Page 12 — Recommendation system (Part B): ALS + Content + Hybrid."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src import recommender as rec
from dashboard.components.bootstrap import bootstrap

st.set_page_config(page_title="Recommendations", page_icon="🤝", layout="wide")
bootstrap(require_data=False)
st.title("Recommendation System (Part B)")
st.caption("Spark MLlib **ALS** (collaborative) + **Content-based** (TF-IDF) + **Hybrid** weighted blend.")


# ---------------------------------------------------------------------------
# Cached state
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def load_interactions() -> pd.DataFrame:
    if rec.INTERACTIONS_PATH.exists():
        return pd.read_parquet(rec.INTERACTIONS_PATH)
    return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_catalog() -> pd.DataFrame:
    if rec.PRODUCT_CATALOG_PATH.exists():
        return pd.read_parquet(rec.PRODUCT_CATALOG_PATH)
    return pd.DataFrame()


# Guard: artefacts must exist
artefact_missing = []
if not rec.ALS_USER_FACTORS_PATH.exists():
    artefact_missing.append("ALS user factors")
if not rec.ALS_ITEM_FACTORS_PATH.exists():
    artefact_missing.append("ALS item factors")
if not rec.CONTENT_PATH.exists():
    artefact_missing.append("Content TF-IDF model")
if not rec.INTERACTIONS_PATH.exists():
    artefact_missing.append("interactions parquet")

if artefact_missing:
    st.error(
        "Recommender artefacts not built yet. From a terminal in the project root run:\n\n"
        "```powershell\npython run_spark_pipeline.py\n```\n\n"
        f"Missing: {', '.join(artefact_missing)}"
    )
    st.stop()


interactions = load_interactions()
catalog = load_catalog()

c1, c2, c3 = st.columns(3)
c1.metric("Customers", f"{interactions['customer_id'].nunique():,}")
c2.metric("Products", f"{len(catalog):,}")
c3.metric("Interactions", f"{len(interactions):,}")

st.divider()

mode = st.radio(
    "Choose a recommendation strategy",
    ["Collaborative (ALS)", "Content-based (similar products)",
     "Content-based (for a customer)", "Hybrid"],
    horizontal=True,
)

top_n = st.slider("Top-N recommendations", 5, 30, 10)

# ---------------------------------------------------------------------------
# Mode dispatch
# ---------------------------------------------------------------------------
if mode == "Collaborative (ALS)":
    customer_options = (
        interactions.groupby("customer_id")["purchases"].sum()
        .sort_values(ascending=False).head(500).index.tolist()
    )
    customer = st.selectbox("Customer", customer_options,
                            help="Top 500 most-active customers.")
    if st.button("Recommend with ALS", width="stretch"):
        with st.spinner("Running ALS inference…"):
            recs = rec.recommend_als(None, customer, top_n=top_n)
        if recs.empty:
            st.warning("No recommendations available for this user.")
        else:
            recs = recs.merge(catalog, on="product_id", how="left", suffixes=("", "_cat"))
            for c in ("product_name", "category", "sub_category", "brand"):
                if c not in recs.columns and f"{c}_cat" in recs.columns:
                    recs[c] = recs[f"{c}_cat"]
            cols = [c for c in
                    ["product_id", "product_name", "category", "sub_category", "brand", "score"]
                    if c in recs.columns]
            st.dataframe(recs[cols], width="stretch", height=420)


elif mode == "Content-based (similar products)":
    product = st.selectbox("Product", catalog["product_id"].tolist(),
                           format_func=lambda pid: f"{pid} — "
                                                   f"{catalog.loc[catalog['product_id']==pid, 'product_name'].iloc[0]}")
    if st.button("Find similar products", width="stretch"):
        with st.spinner("Computing content similarity…"):
            recs = rec.recommend_content_for_product(product, top_n=top_n)
        if recs.empty:
            st.warning("Product not found in catalog.")
        else:
            st.dataframe(recs, width="stretch", height=420)


elif mode == "Content-based (for a customer)":
    customer_options = (
        interactions.groupby("customer_id")["purchases"].sum()
        .sort_values(ascending=False).head(500).index.tolist()
    )
    customer = st.selectbox("Customer", customer_options)
    if st.button("Recommend by content profile", width="stretch"):
        with st.spinner("Building user profile + scoring catalog…"):
            recs = rec.recommend_content_for_user(customer, interactions, top_n=top_n)
        if recs.empty:
            st.warning("No purchases recorded for this customer.")
        else:
            st.dataframe(recs, width="stretch", height=420)


else:  # Hybrid
    customer_options = (
        interactions.groupby("customer_id")["purchases"].sum()
        .sort_values(ascending=False).head(500).index.tolist()
    )
    customer = st.selectbox("Customer", customer_options)
    alpha = st.slider("Hybrid weight (collab vs content)", 0.0, 1.0, 0.6, 0.05,
                       help="0 = pure content-based, 1 = pure collaborative.")
    if st.button("Recommend (hybrid)", width="stretch"):
        with st.spinner("Hybrid scoring…"):
            recs = rec.recommend_hybrid(None, customer, interactions,
                                          top_n=top_n, alpha=alpha)
        if recs.empty:
            st.warning("No recommendations available for this user.")
        else:
            st.dataframe(recs, width="stretch", height=420)


with st.expander("How does each strategy work?"):
    st.markdown(
        "- **ALS** (Spark MLlib): factorizes a user × product purchase matrix into "
        "latent embeddings. Good for users with purchase history; struggles on cold-start users.\n"
        "- **Content-based**: TF-IDF over `product_name + category + sub_category + brand`. "
        "Cosine similarity ranks the catalog. Cold-start safe at the product level.\n"
        "- **Hybrid**: weighted blend. `score = alpha * ALS + (1-alpha) * content`."
    )
