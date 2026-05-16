"""Train / persist / load wrappers for all models used in the assignment.

Kept intentionally simple: scikit-learn pipelines + joblib persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    GradientBoostingRegressor,
    HistGradientBoostingClassifier,
)
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config
from .evaluation import (
    ClassificationReport,
    RegressionReport,
    evaluate_classifier,
    evaluate_clustering,
    evaluate_regressor,
)
from .features import build_model_frame, feature_columns
from .utils import get_logger, timer

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Preprocessing pipeline
# ---------------------------------------------------------------------------
def _build_preprocessor(num_cols: list[str], cat_cols: list[str]) -> ColumnTransformer:
    numeric = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False, max_categories=30)
    except TypeError:
        # sklearn < 1.2 fallback
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
    categorical = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("ohe", ohe),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", numeric, num_cols),
            ("cat", categorical, cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


# ---------------------------------------------------------------------------
# Classifier (default target: is_returned)
# ---------------------------------------------------------------------------
@dataclass
class TrainedClassifier:
    pipeline: Pipeline
    target: str
    num_cols: list[str]
    cat_cols: list[str]
    report: ClassificationReport
    feature_importances_: pd.Series | None = None


def train_classifier(df: pd.DataFrame, target: str = "is_returned") -> TrainedClassifier:
    if target not in df.columns:
        raise KeyError(f"Target column missing: {target}")

    num_cols, cat_cols = feature_columns(df, target=target)
    X, y = build_model_frame(df, target=target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE,
        stratify=y if y.nunique() > 1 else None,
    )

    pre = _build_preprocessor(num_cols, cat_cols)
    # HistGradientBoostingClassifier supports class_weight (sklearn >= 1.4),
    # which is essential for the imbalanced is_returned target.
    try:
        clf_est = HistGradientBoostingClassifier(
            random_state=config.RANDOM_STATE,
            max_iter=200,
            learning_rate=0.08,
            max_depth=6,
            class_weight="balanced",
        )
    except TypeError:
        clf_est = HistGradientBoostingClassifier(
            random_state=config.RANDOM_STATE, max_iter=200, learning_rate=0.08, max_depth=6,
        )

    pipe = Pipeline([
        ("pre", pre),
        ("clf", clf_est),
    ])

    with timer(f"train classifier({target})"):
        pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    try:
        y_proba = pipe.predict_proba(X_test)[:, 1]
    except Exception:
        y_proba = None
    report = evaluate_classifier(y_test, y_pred, y_proba=y_proba)
    log.info("Classifier(%s) -> %s", target, report.to_dict())

    fi = None
    try:
        ohe_features = pipe.named_steps["pre"].get_feature_names_out()
        importances = pipe.named_steps["clf"].feature_importances_
        fi = pd.Series(importances, index=ohe_features).sort_values(ascending=False)
    except Exception as exc:  # pragma: no cover - defensive
        log.debug("feature_importances unavailable: %s", exc)

    return TrainedClassifier(
        pipeline=pipe,
        target=target,
        num_cols=num_cols,
        cat_cols=cat_cols,
        report=report,
        feature_importances_=fi,
    )


# ---------------------------------------------------------------------------
# Regressor (default target: profit_usd)
# ---------------------------------------------------------------------------
@dataclass
class TrainedRegressor:
    pipeline: Pipeline
    target: str
    num_cols: list[str]
    cat_cols: list[str]
    report: RegressionReport


def train_regressor(df: pd.DataFrame, target: str = "profit_usd") -> TrainedRegressor:
    num_cols, cat_cols = feature_columns(df, target=target)
    # Avoid leakage: don't use total_price/cost when predicting profit.
    leaky = {"profit_usd", "cost_usd", "total_price_usd", "discount_amount_usd", "tax_usd", "profit_margin_percent"}
    num_cols = [c for c in num_cols if c not in leaky]

    X, y = build_model_frame(df, target=target, extra_drop=leaky)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE
    )

    pre = _build_preprocessor(num_cols, cat_cols)
    pipe = Pipeline([
        ("pre", pre),
        ("reg", GradientBoostingRegressor(random_state=config.RANDOM_STATE, n_estimators=150, max_depth=3)),
    ])
    with timer(f"train regressor({target})"):
        pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    report = evaluate_regressor(y_test, y_pred)
    log.info("Regressor(%s) -> %s", target, report.to_dict())

    return TrainedRegressor(
        pipeline=pipe, target=target, num_cols=num_cols, cat_cols=cat_cols, report=report
    )


# ---------------------------------------------------------------------------
# Clustering (RFM)
# ---------------------------------------------------------------------------
@dataclass
class TrainedClustering:
    model: KMeans
    scaler: StandardScaler
    metrics: dict[str, float]
    n_clusters: int


def train_clustering(rfm: pd.DataFrame, n_clusters: int = 5) -> TrainedClustering:
    feats = rfm[["recency", "frequency", "monetary"]].astype(float)
    scaler = StandardScaler()
    X = scaler.fit_transform(feats)
    km = KMeans(n_clusters=n_clusters, n_init=10, random_state=config.RANDOM_STATE)
    with timer(f"train kmeans(k={n_clusters})"):
        labels = km.fit_predict(X)
    metrics = evaluate_clustering(X, labels)
    log.info("Clustering -> %s", metrics)
    return TrainedClustering(model=km, scaler=scaler, metrics=metrics, n_clusters=n_clusters)


# ---------------------------------------------------------------------------
# Time-series forecasting (Holt-Winters, robust on Windows + py 3.12)
# ---------------------------------------------------------------------------
@dataclass
class TrainedForecaster:
    fitted: Any
    history: pd.DataFrame
    metric_mae: float


def train_forecaster(daily: pd.DataFrame, value_col: str = "revenue") -> TrainedForecaster:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing

    s = (
        daily[["date", value_col]]
        .dropna()
        .sort_values("date")
        .set_index("date")[value_col]
        .asfreq("D")
        .interpolate()
    )

    if len(s) < 30:
        raise ValueError("Not enough daily history to fit a forecaster.")

    train, test = s.iloc[:-14], s.iloc[-14:]
    seasonal_periods = 7
    with timer("train forecaster (Holt-Winters)"):
        model = ExponentialSmoothing(
            train,
            trend="add",
            seasonal="add" if len(train) >= 2 * seasonal_periods else None,
            seasonal_periods=seasonal_periods if len(train) >= 2 * seasonal_periods else None,
            initialization_method="estimated",
        )
        fit = model.fit(optimized=True)
    pred = fit.forecast(len(test))
    mae = float((pred - test).abs().mean())
    history = pd.DataFrame({"date": s.index, value_col: s.values}).reset_index(drop=True)
    log.info("Forecaster MAE on holdout (14d): %.2f", mae)
    return TrainedForecaster(fitted=fit, history=history, metric_mae=mae)


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------
def save(obj: Any, path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, path)
    log.info("Saved %s", path)


def load(path):
    return joblib.load(path)


__all__ = [
    "TrainedClassifier",
    "TrainedRegressor",
    "TrainedClustering",
    "TrainedForecaster",
    "train_classifier",
    "train_regressor",
    "train_clustering",
    "train_forecaster",
    "save",
    "load",
]
