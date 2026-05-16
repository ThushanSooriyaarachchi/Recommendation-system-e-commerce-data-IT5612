"""Recommendation systems for Part B of the Big Data assignment.

Implements three approaches:

1. **Collaborative filtering** with Spark MLlib **ALS** (matrix factorization
   on implicit user-item interactions inferred from purchase frequency).

2. **Content-based** filtering with sklearn TF-IDF over product text features
   (product_name + category + sub_category + brand).

3. **Hybrid** weighted blend of the two scores.

All three return a uniform interface: `recommend(user_id, top_n)` -> list[dict].
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from . import config
from .utils import get_logger, timer

log = get_logger(__name__)


REC_DIR: Path = config.MODELS_DIR / "recommender"
ALS_MODEL_DIR: Path = REC_DIR / "als"
ALS_USER_FACTORS_PATH: Path = REC_DIR / "als_user_factors.parquet"
ALS_ITEM_FACTORS_PATH: Path = REC_DIR / "als_item_factors.parquet"
ALS_META_PATH: Path = REC_DIR / "als_meta.json"
CONTENT_PATH: Path = REC_DIR / "content.joblib"
USER_INDEX_PATH: Path = REC_DIR / "user_index.parquet"
ITEM_INDEX_PATH: Path = REC_DIR / "item_index.parquet"
PRODUCT_CATALOG_PATH: Path = REC_DIR / "product_catalog.parquet"
INTERACTIONS_PATH: Path = REC_DIR / "interactions.parquet"


# ---------------------------------------------------------------------------
# Interaction matrix
# ---------------------------------------------------------------------------
def build_interactions(pdf: pd.DataFrame) -> pd.DataFrame:
    """Aggregate purchase counts per (customer, product). Used by ALS."""
    inter = (
        pdf.groupby(["customer_id", "product_id"], as_index=False)
        .agg(purchases=("order_id", "nunique"),
             revenue=("total_price_usd", "sum"),
             rating=("rating", "mean"))
    )
    # Implicit feedback: confidence = log(1 + purchases) * mean_rating
    inter["score"] = np.log1p(inter["purchases"]) * (inter["rating"].fillna(3.0) / 5.0)
    return inter


def build_product_catalog(pdf: pd.DataFrame) -> pd.DataFrame:
    """One row per product with text features used by the content recommender."""
    catalog = (
        pdf.groupby("product_id", as_index=False)
        .agg(product_name=("product_name", "first"),
             category=("category", "first"),
             sub_category=("sub_category", "first"),
             brand=("brand", "first"),
             unit_price_usd=("unit_price_usd", "mean"),
             avg_rating=("rating", "mean"),
             reviews_count=("product_reviews_count", "max"))
    )
    catalog["text_features"] = (
        catalog["product_name"].fillna("") + " "
        + catalog["category"].astype(str).fillna("") + " "
        + catalog["sub_category"].astype(str).fillna("") + " "
        + catalog["brand"].astype(str).fillna("")
    ).str.strip()
    return catalog


# ---------------------------------------------------------------------------
# ALS (Spark MLlib)
# ---------------------------------------------------------------------------
@dataclass
class ALSArtifacts:
    model_path: Path
    user_index: pd.DataFrame  # customer_id, user_idx
    item_index: pd.DataFrame  # product_id, item_idx
    rmse: float


def train_als(spark, interactions_pdf: pd.DataFrame) -> ALSArtifacts:
    """Train Spark ALS on the interaction table.

    The interactions DataFrame must have columns: customer_id, product_id, score.
    """
    from pyspark.ml.recommendation import ALS
    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.sql import functions as F
    from pyspark.sql.types import StringType

    REC_DIR.mkdir(parents=True, exist_ok=True)
    log.info("ALS: training on %s interactions", f"{len(interactions_pdf):,}")

    # Build integer user/item indices (ALS requires ints).
    users = (
        interactions_pdf[["customer_id"]].drop_duplicates().reset_index(drop=True)
        .reset_index(names="user_idx")
    )
    items = (
        interactions_pdf[["product_id"]].drop_duplicates().reset_index(drop=True)
        .reset_index(names="item_idx")
    )
    inter = (
        interactions_pdf.merge(users, on="customer_id")
        .merge(items, on="product_id")
        [["user_idx", "item_idx", "score"]]
    )

    sdf = spark.createDataFrame(inter)
    train, test = sdf.randomSplit([0.85, 0.15], seed=42)

    als = ALS(
        userCol="user_idx", itemCol="item_idx", ratingCol="score",
        rank=24, regParam=0.08, maxIter=10,
        coldStartStrategy="drop", implicitPrefs=True,
        nonnegative=True, seed=42,
    )
    with timer("ALS fit"):
        model = als.fit(train)
    pred = model.transform(test)
    rmse = RegressionEvaluator(metricName="rmse", labelCol="score",
                                predictionCol="prediction").evaluate(pred)
    log.info("ALS RMSE on holdout: %.4f", rmse)

    # Export user/item latent factors to parquet so inference can run
    # without a live Spark session (avoids Hadoop winutils dependency on Windows).
    with timer("ALS extract factors"):
        user_factors = (
            model.userFactors
            .withColumnRenamed("id", "user_idx")
            .withColumnRenamed("features", "factors")
            .toPandas()
        )
        item_factors = (
            model.itemFactors
            .withColumnRenamed("id", "item_idx")
            .withColumnRenamed("features", "factors")
            .toPandas()
        )

    REC_DIR.mkdir(parents=True, exist_ok=True)
    # Convert array column to a 2-D numpy array stored as list-of-lists in parquet.
    user_factors["factors"] = user_factors["factors"].apply(list)
    item_factors["factors"] = item_factors["factors"].apply(list)
    user_factors.to_parquet(ALS_USER_FACTORS_PATH, index=False)
    item_factors.to_parquet(ALS_ITEM_FACTORS_PATH, index=False)

    users.to_parquet(USER_INDEX_PATH, index=False)
    items.to_parquet(ITEM_INDEX_PATH, index=False)

    meta = {"rank": int(model.rank), "rmse": float(rmse),
            "n_users": int(len(user_factors)), "n_items": int(len(item_factors))}
    ALS_META_PATH.write_text(json.dumps(meta, indent=2))
    log.info("ALS factors saved (%s users, %s items, rank=%s)",
             meta["n_users"], meta["n_items"], meta["rank"])

    return ALSArtifacts(model_path=ALS_USER_FACTORS_PATH, user_index=users,
                        item_index=items, rmse=rmse)


@dataclass
class _ALSCache:
    user_idx_map: dict[str, int]
    item_factors_idx: np.ndarray  # (n_items,)
    item_factors_mat: np.ndarray  # (n_items, rank)
    user_factors_idx_map: dict[int, int]  # user_idx -> row in user matrix
    user_factors_mat: np.ndarray  # (n_users, rank)
    items_meta: pd.DataFrame  # product_id, item_idx + catalog cols


_als_cache: _ALSCache | None = None


def _load_als_cache() -> _ALSCache:
    """Load ALS factor matrices once and keep them in-process."""
    global _als_cache
    if _als_cache is not None:
        return _als_cache
    if not ALS_USER_FACTORS_PATH.exists() or not ALS_ITEM_FACTORS_PATH.exists():
        raise FileNotFoundError("ALS factors not trained — run run_spark_pipeline.py.")

    users = pd.read_parquet(USER_INDEX_PATH)
    items = pd.read_parquet(ITEM_INDEX_PATH)
    user_fac = pd.read_parquet(ALS_USER_FACTORS_PATH)
    item_fac = pd.read_parquet(ALS_ITEM_FACTORS_PATH)

    # Stack the list-of-floats column into a 2D numpy matrix.
    user_mat = np.vstack(user_fac["factors"].apply(np.asarray).to_numpy()).astype(np.float32)
    item_mat = np.vstack(item_fac["factors"].apply(np.asarray).to_numpy()).astype(np.float32)

    user_idx_map = dict(zip(users["customer_id"].astype(str), users["user_idx"].astype(int)))
    user_factors_idx_map = dict(zip(user_fac["user_idx"].astype(int),
                                    np.arange(len(user_fac), dtype=int)))

    items_meta = items.copy()
    items_meta["row"] = items_meta["item_idx"].map(
        dict(zip(item_fac["item_idx"].astype(int), np.arange(len(item_fac), dtype=int)))
    )

    _als_cache = _ALSCache(
        user_idx_map=user_idx_map,
        item_factors_idx=item_fac["item_idx"].astype(int).to_numpy(),
        item_factors_mat=item_mat,
        user_factors_idx_map=user_factors_idx_map,
        user_factors_mat=user_mat,
        items_meta=items_meta,
    )
    return _als_cache


def recommend_als(_spark_unused, customer_id: str, top_n: int = 10) -> pd.DataFrame:
    """Top-N ALS recommendations using exported latent factors.

    `_spark_unused` is kept to preserve the public API; inference is now
    pure NumPy and does not need a Spark session.
    """
    cache = _load_als_cache()
    customer_id = str(customer_id)
    if customer_id not in cache.user_idx_map:
        return pd.DataFrame(columns=["product_id", "score"])

    user_idx = cache.user_idx_map[customer_id]
    if user_idx not in cache.user_factors_idx_map:
        return pd.DataFrame(columns=["product_id", "score"])

    row = cache.user_factors_idx_map[user_idx]
    user_vec = cache.user_factors_mat[row]                           # (rank,)
    scores = cache.item_factors_mat @ user_vec                       # (n_items,)

    n = min(top_n, scores.shape[0])
    top_rows = np.argpartition(-scores, n - 1)[:n]
    top_rows = top_rows[np.argsort(-scores[top_rows])]
    top_item_idx = cache.item_factors_idx[top_rows]
    top_scores = scores[top_rows]

    out = pd.DataFrame({"item_idx": top_item_idx, "score": top_scores.astype(float)})
    out = out.merge(cache.items_meta.drop(columns=["row"]), on="item_idx").drop(columns=["item_idx"])
    return out.sort_values("score", ascending=False).reset_index(drop=True).head(top_n)


# ---------------------------------------------------------------------------
# Content-based (sklearn TF-IDF)
# ---------------------------------------------------------------------------
@dataclass
class ContentArtifacts:
    vectorizer: Any
    matrix: Any  # scipy.sparse
    catalog: pd.DataFrame


def train_content(catalog: pd.DataFrame) -> ContentArtifacts:
    from sklearn.feature_extraction.text import TfidfVectorizer

    REC_DIR.mkdir(parents=True, exist_ok=True)
    catalog = catalog.copy()
    catalog["text_features"] = catalog["text_features"].fillna("")

    vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_features=20_000, min_df=2)
    with timer("Content TF-IDF fit"):
        matrix = vec.fit_transform(catalog["text_features"])
    arts = ContentArtifacts(vectorizer=vec, matrix=matrix, catalog=catalog.reset_index(drop=True))
    joblib.dump(arts, CONTENT_PATH)
    catalog.to_parquet(PRODUCT_CATALOG_PATH, index=False)
    log.info("Content recommender saved (%s products, %s features)",
             len(catalog), matrix.shape[1])
    return arts


def _load_content() -> ContentArtifacts:
    return joblib.load(CONTENT_PATH)


def recommend_content_for_product(product_id: str, top_n: int = 10) -> pd.DataFrame:
    """Most similar products to a given product (content-based)."""
    from sklearn.metrics.pairwise import cosine_similarity

    arts = _load_content()
    cat = arts.catalog
    if product_id not in set(cat["product_id"]):
        return pd.DataFrame(columns=["product_id", "score"])

    idx = cat.index[cat["product_id"] == product_id][0]
    sims = cosine_similarity(arts.matrix[idx], arts.matrix).ravel()
    sims[idx] = -1
    top_idx = sims.argsort()[::-1][:top_n]
    out = cat.iloc[top_idx][["product_id", "product_name", "category", "sub_category", "brand"]].copy()
    out["score"] = sims[top_idx]
    return out.reset_index(drop=True)


def recommend_content_for_user(customer_id: str, interactions: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
    """Aggregate user's purchased products into a profile vector, then rank catalog."""
    from sklearn.metrics.pairwise import cosine_similarity

    arts = _load_content()
    cat = arts.catalog

    purchases = interactions[interactions["customer_id"] == customer_id]
    if purchases.empty:
        return pd.DataFrame(columns=["product_id", "score"])

    bought_idx = cat.index[cat["product_id"].isin(purchases["product_id"])].tolist()
    if not bought_idx:
        return pd.DataFrame(columns=["product_id", "score"])

    profile = arts.matrix[bought_idx].mean(axis=0)
    profile = np.asarray(profile)
    sims = cosine_similarity(profile.reshape(1, -1), arts.matrix).ravel()
    # Don't recommend already-purchased items.
    sims[bought_idx] = -1
    top_idx = sims.argsort()[::-1][:top_n]
    out = cat.iloc[top_idx][["product_id", "product_name", "category", "sub_category", "brand"]].copy()
    out["score"] = sims[top_idx]
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Hybrid
# ---------------------------------------------------------------------------
def recommend_hybrid(spark, customer_id: str, interactions: pd.DataFrame,
                     top_n: int = 10, alpha: float = 0.6) -> pd.DataFrame:
    """Hybrid score = alpha * ALS_score + (1 - alpha) * content_score (both min-max scaled)."""
    als = recommend_als(spark, customer_id, top_n=max(50, top_n * 5))
    content = recommend_content_for_user(customer_id, interactions, top_n=max(50, top_n * 5))

    if als.empty and content.empty:
        return pd.DataFrame(columns=["product_id", "score"])

    def _scale(s: pd.Series) -> pd.Series:
        if s.empty:
            return s
        lo, hi = float(s.min()), float(s.max())
        return (s - lo) / (hi - lo) if hi > lo else s * 0 + 1.0

    if not als.empty:
        als = als.assign(als_score=_scale(als["score"])).drop(columns=["score"])
    if not content.empty:
        content = content.assign(content_score=_scale(content["score"])).drop(columns=["score"])

    merged = pd.merge(als, content, on="product_id", how="outer")
    merged["als_score"] = merged.get("als_score", 0).fillna(0)
    merged["content_score"] = merged.get("content_score", 0).fillna(0)
    merged["score"] = alpha * merged["als_score"] + (1 - alpha) * merged["content_score"]

    # Bring back any missing metadata.
    arts = _load_content()
    cat = arts.catalog[["product_id", "product_name", "category", "sub_category", "brand"]]
    merged = merged.merge(cat, on="product_id", how="left", suffixes=("", "_cat"))
    for c in ("product_name", "category", "sub_category", "brand"):
        if c not in merged.columns and f"{c}_cat" in merged.columns:
            merged[c] = merged[f"{c}_cat"]
    cols = ["product_id", "product_name", "category", "sub_category", "brand",
            "als_score", "content_score", "score"]
    out = merged[cols].sort_values("score", ascending=False).head(top_n).reset_index(drop=True)
    return out


__all__ = [
    "build_interactions",
    "build_product_catalog",
    "train_als",
    "train_content",
    "recommend_als",
    "recommend_content_for_product",
    "recommend_content_for_user",
    "recommend_hybrid",
    "REC_DIR",
    "ALS_USER_FACTORS_PATH",
    "ALS_ITEM_FACTORS_PATH",
    "ALS_META_PATH",
    "CONTENT_PATH",
    "USER_INDEX_PATH",
    "ITEM_INDEX_PATH",
    "PRODUCT_CATALOG_PATH",
    "INTERACTIONS_PATH",
]
