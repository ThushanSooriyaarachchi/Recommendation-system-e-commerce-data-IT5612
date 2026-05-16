"""Lightweight metric helpers for classification, regression, clustering."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)


@dataclass
class ClassificationReport:
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float | None = None
    extras: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            **self.extras,
        }


@dataclass
class RegressionReport:
    mae: float
    rmse: float
    r2: float

    def to_dict(self) -> dict[str, Any]:
        return {"mae": self.mae, "rmse": self.rmse, "r2": self.r2}


def evaluate_classifier(y_true, y_pred, y_proba=None) -> ClassificationReport:
    rep = ClassificationReport(
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
    )
    if y_proba is not None:
        try:
            rep.roc_auc = float(roc_auc_score(y_true, y_proba))
        except Exception:
            rep.roc_auc = None
    return rep


def evaluate_regressor(y_true, y_pred) -> RegressionReport:
    mse = float(mean_squared_error(y_true, y_pred))
    return RegressionReport(
        mae=float(mean_absolute_error(y_true, y_pred)),
        rmse=float(np.sqrt(mse)),
        r2=float(r2_score(y_true, y_pred)),
    )


def evaluate_clustering(X, labels) -> dict[str, float]:
    out: dict[str, float] = {}
    unique = set(labels)
    if len(unique) > 1 and -1 not in unique:
        try:
            out["silhouette"] = float(silhouette_score(X, labels, sample_size=10_000, random_state=0))
        except Exception:
            pass
    out["n_clusters"] = float(len(unique - {-1}))
    return out


__all__ = [
    "ClassificationReport",
    "RegressionReport",
    "evaluate_classifier",
    "evaluate_regressor",
    "evaluate_clustering",
]
