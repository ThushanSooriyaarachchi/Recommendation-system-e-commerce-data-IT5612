"""Spark MLlib classifier + regressor (mirrors src/models.py at scale)."""

from __future__ import annotations

from dataclasses import dataclass

from pyspark.ml import Pipeline as SparkPipeline
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import (
    BinaryClassificationEvaluator,
    MulticlassClassificationEvaluator,
    RegressionEvaluator,
)
from pyspark.ml.feature import Imputer, OneHotEncoder, StringIndexer, VectorAssembler
from pyspark.sql import DataFrame

from .features import CATEGORICAL_FEATURES, NUMERIC_FEATURES
from .utils import get_logger, timer

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------
def _existing(df: DataFrame, cols: list[str]) -> list[str]:
    return [c for c in cols if c in df.columns]


def _build_feature_pipeline(
    df: DataFrame,
    target: str,
    extra_drop: tuple[str, ...] = (),
) -> tuple[list, list[str], list[str]]:
    drop = set(extra_drop) | {target}
    num_cols = [c for c in _existing(df, NUMERIC_FEATURES) if c not in drop]
    cat_cols = [c for c in _existing(df, CATEGORICAL_FEATURES) if c not in drop]

    stages: list = []
    imputed_cols = [f"{c}__i" for c in num_cols]
    if num_cols:
        stages.append(Imputer(inputCols=num_cols, outputCols=imputed_cols, strategy="median"))

    indexed_cols = [f"{c}__idx" for c in cat_cols]
    encoded_cols = [f"{c}__oh" for c in cat_cols]
    for c, ic in zip(cat_cols, indexed_cols):
        stages.append(StringIndexer(inputCol=c, outputCol=ic, handleInvalid="keep"))
    if cat_cols:
        stages.append(OneHotEncoder(inputCols=indexed_cols, outputCols=encoded_cols, handleInvalid="keep"))

    feature_cols = imputed_cols + encoded_cols
    stages.append(VectorAssembler(inputCols=feature_cols, outputCol="features", handleInvalid="keep"))
    return stages, num_cols, cat_cols


# ---------------------------------------------------------------------------
# Classifier (target: is_returned)
# ---------------------------------------------------------------------------
@dataclass
class SparkClassifierResult:
    model: object
    target: str
    metrics: dict[str, float]


def train_spark_classifier(df: DataFrame, target: str = "is_returned") -> SparkClassifierResult:
    if target not in df.columns:
        raise KeyError(f"Target column missing: {target}")

    df = df.dropna(subset=[target]).withColumn(target, df[target].cast("int"))
    train, test = df.randomSplit([0.8, 0.2], seed=42)

    stages, _, _ = _build_feature_pipeline(df, target=target)
    gbt = GBTClassifier(featuresCol="features", labelCol=target, maxIter=40, maxDepth=5, seed=42)
    pipeline = SparkPipeline(stages=stages + [gbt])

    with timer(f"spark train classifier({target})"):
        model = pipeline.fit(train)

    pred = model.transform(test)
    auc = BinaryClassificationEvaluator(labelCol=target, rawPredictionCol="rawPrediction",
                                         metricName="areaUnderROC").evaluate(pred)
    f1 = MulticlassClassificationEvaluator(labelCol=target, predictionCol="prediction",
                                            metricName="f1").evaluate(pred)
    acc = MulticlassClassificationEvaluator(labelCol=target, predictionCol="prediction",
                                             metricName="accuracy").evaluate(pred)
    metrics = {"roc_auc": auc, "f1": f1, "accuracy": acc}
    log.info("Spark classifier(%s) -> %s", target, metrics)
    return SparkClassifierResult(model=model, target=target, metrics=metrics)


# ---------------------------------------------------------------------------
# Regressor (target: profit_usd)
# ---------------------------------------------------------------------------
@dataclass
class SparkRegressorResult:
    model: object
    target: str
    metrics: dict[str, float]


def train_spark_regressor(df: DataFrame, target: str = "profit_usd") -> SparkRegressorResult:
    leaky = ("profit_usd", "cost_usd", "total_price_usd", "discount_amount_usd",
             "tax_usd", "profit_margin_percent")
    df = df.dropna(subset=[target]).withColumn(target, df[target].cast("double"))
    train, test = df.randomSplit([0.8, 0.2], seed=42)

    stages, _, _ = _build_feature_pipeline(df, target=target, extra_drop=leaky)
    gbt = GBTRegressor(featuresCol="features", labelCol=target, maxIter=60, maxDepth=5, seed=42)
    pipeline = SparkPipeline(stages=stages + [gbt])

    with timer(f"spark train regressor({target})"):
        model = pipeline.fit(train)

    pred = model.transform(test)
    rmse = RegressionEvaluator(labelCol=target, predictionCol="prediction",
                                metricName="rmse").evaluate(pred)
    mae = RegressionEvaluator(labelCol=target, predictionCol="prediction",
                               metricName="mae").evaluate(pred)
    r2 = RegressionEvaluator(labelCol=target, predictionCol="prediction",
                              metricName="r2").evaluate(pred)
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    log.info("Spark regressor(%s) -> %s", target, metrics)
    return SparkRegressorResult(model=model, target=target, metrics=metrics)


__all__ = [
    "SparkClassifierResult",
    "SparkRegressorResult",
    "train_spark_classifier",
    "train_spark_regressor",
]
