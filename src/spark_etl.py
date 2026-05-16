"""PySpark equivalent of the pandas ETL: load → clean → aggregate → write.

Outputs the SAME parquet files used by the Streamlit dashboard, so the
dashboard can be served from either the pandas pipeline or the Spark
pipeline interchangeably.
"""

from __future__ import annotations

from pathlib import Path

from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType

from . import config
from .utils import get_logger, timer

log = get_logger(__name__)


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
def load_csv(spark: SparkSession, csv_path: Path | None = None) -> DataFrame:
    """Read the raw CSV into a Spark DataFrame with inferred schema."""
    csv_path = csv_path or config.RAW_CSV
    log.info("Spark reading %s", csv_path)
    return (
        spark.read
        .option("header", True)
        .option("inferSchema", True)
        .option("mode", "DROPMALFORMED")
        .csv(str(csv_path))
    )


# ---------------------------------------------------------------------------
# Clean
# ---------------------------------------------------------------------------
def clean(df: DataFrame) -> DataFrame:
    """Mirror of `src.preprocessing.clean` for Spark."""
    with timer("spark clean"):
        df = (
            df
            .withColumn("order_date", F.to_timestamp("order_date"))
            .withColumn("account_creation_date", F.to_date("account_creation_date"))
            .filter(F.col("order_id").isNotNull())
            .filter(F.col("customer_id").isNotNull())
            .filter(F.col("order_date").isNotNull())
            .filter(F.col("total_price_usd") >= 0)
        )

        df = (
            df
            .withColumn("order_year", F.year("order_date").cast(IntegerType()))
            .withColumn("order_month", F.month("order_date").cast(IntegerType()))
            .withColumn("order_day", F.dayofmonth("order_date").cast(IntegerType()))
            .withColumn("order_hour", F.hour("order_date").cast(IntegerType()))
            .withColumn("order_dow", F.dayofweek("order_date").cast(IntegerType()) - 1)
            .withColumn("order_week", F.weekofyear("order_date").cast(IntegerType()))
            .withColumn("order_yearmonth", F.date_format("order_date", "yyyy-MM"))
            .withColumn("is_weekend_bool", (F.col("order_dow") >= 5))
            .withColumn(
                "customer_tenure_days",
                F.greatest(
                    F.lit(0),
                    F.datediff(F.col("order_date").cast("date"), F.col("account_creation_date")),
                ),
            )
        )

        for col, lo, hi in [
            ("session_duration_minutes", 0, 300),
            ("delivery_days", 0, 60),
            ("age", 0, 100),
            ("discount_percent", 0, 90),
            ("fraud_risk_score", 0, 100),
            ("profit_margin_percent", -100, 100),
        ]:
            if col in df.columns:
                df = df.withColumn(col, F.least(F.greatest(F.col(col), F.lit(lo)), F.lit(hi)))

        df = (
            df
            .fillna({"discount_percent": 0, "discount_amount_usd": 0, "tax_usd": 0})
            .fillna({"return_reason": "Not Returned", "customer_feedback": "", "coupon_code": ""})
        )

        df = (
            df
            .withColumn("is_returned", (F.col("order_status") == "Returned").cast("int"))
            .withColumn("is_high_fraud", (F.col("fraud_risk_score") >= 75).cast("int"))
            .withColumn("has_support_ticket", (F.col("support_ticket_created") == "Yes").cast("int"))
            .withColumn("had_abandoned_cart", (F.col("abandoned_cart_before") == "Yes").cast("int"))
            .withColumn("coupon_used_bool", (F.col("coupon_used") == "Yes").cast("int"))
        )
    return df


# ---------------------------------------------------------------------------
# Aggregations
# ---------------------------------------------------------------------------
def kpi_daily(df: DataFrame) -> DataFrame:
    return (
        df.groupBy(F.to_date("order_date").alias("date"))
        .agg(
            F.sum("total_price_usd").alias("revenue"),
            F.sum("profit_usd").alias("profit"),
            F.countDistinct("order_id").alias("orders"),
            F.countDistinct("customer_id").alias("customers"),
            F.avg("total_price_usd").alias("avg_basket"),
            F.avg("is_returned").alias("return_rate"),
        )
        .orderBy("date")
    )


def kpi_country(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("country")
        .agg(
            F.sum("total_price_usd").alias("revenue"),
            F.sum("profit_usd").alias("profit"),
            F.countDistinct("order_id").alias("orders"),
            F.countDistinct("customer_id").alias("customers"),
            F.avg("is_returned").alias("return_rate"),
        )
        .orderBy(F.col("revenue").desc())
    )


def kpi_category(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("category", "sub_category")
        .agg(
            F.sum("total_price_usd").alias("revenue"),
            F.sum("profit_usd").alias("profit"),
            F.countDistinct("order_id").alias("orders"),
            F.avg("profit_margin_percent").alias("avg_margin"),
            F.avg("rating").alias("avg_rating"),
        )
        .orderBy(F.col("revenue").desc())
    )


def kpi_brand(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("brand")
        .agg(
            F.sum("total_price_usd").alias("revenue"),
            F.sum("profit_usd").alias("profit"),
            F.countDistinct("order_id").alias("orders"),
            F.avg("rating").alias("avg_rating"),
        )
        .orderBy(F.col("revenue").desc())
    )


def kpi_channel(df: DataFrame) -> DataFrame:
    return (
        df.groupBy("campaign_source", "traffic_source", "device_type")
        .agg(
            F.sum("total_price_usd").alias("revenue"),
            F.countDistinct("order_id").alias("orders"),
            F.avg("session_duration_minutes").alias("avg_session_min"),
            F.avg("coupon_used_bool").alias("coupon_share"),
        )
        .orderBy(F.col("revenue").desc())
    )


# ---------------------------------------------------------------------------
# RFM in Spark
# ---------------------------------------------------------------------------
def compute_rfm(df: DataFrame) -> DataFrame:
    snapshot = df.agg(F.max("order_date")).first()[0]
    if snapshot is None:
        raise ValueError("No order_date present.")
    rfm = (
        df.groupBy("customer_id")
        .agg(
            F.datediff(F.lit(snapshot).cast("timestamp"), F.max("order_date")).alias("recency"),
            F.countDistinct("order_id").alias("frequency"),
            F.sum("total_price_usd").alias("monetary"),
        )
    )
    return rfm


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------
def write_pandas_parquet(spark_df: DataFrame, path: Path) -> int:
    """Collect a small Spark DF to pandas + write parquet (compatible with dashboard)."""
    pdf = spark_df.toPandas()
    path.parent.mkdir(parents=True, exist_ok=True)
    pdf.to_parquet(path, index=False)
    return len(pdf)


__all__ = [
    "load_csv",
    "clean",
    "kpi_daily",
    "kpi_country",
    "kpi_category",
    "kpi_brand",
    "kpi_channel",
    "compute_rfm",
    "write_pandas_parquet",
]
