"""Apache Spark pipeline (Part A + Part B of the Big Data assignment).

Steps
-----
1. Read the raw CSV with PySpark
2. Clean + feature engineer using Spark SQL
3. Write the same parquet files used by the Streamlit dashboard
4. Train a Spark MLlib model (Part A)
5. Train ALS recommender + content-based recommender (Part B)

Run:
    python run_spark_pipeline.py [--quick]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from src import config
from src import recommender as rec
from src import spark_etl as setl
from src.spark_models import train_spark_classifier, train_spark_regressor
from src.spark_session import get_spark
from src.utils import get_logger, timer

log = get_logger("run_spark")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true",
                        help="Use a 100k sample for Spark MLlib training (still uses full dataset for ETL).")
    parser.add_argument("--skip-models", action="store_true")
    parser.add_argument("--skip-recsys", action="store_true")
    args = parser.parse_args()

    config.ensure_dirs()
    spark = get_spark("ecom-bigdata")

    with timer("Spark pipeline"):
        # 1+2. Load + clean
        raw = setl.load_csv(spark)
        df = setl.clean(raw).cache()
        n = df.count()
        log.info("Cleaned dataset: %s rows, %s cols", f"{n:,}", len(df.columns))

        # 3. Aggregations -> parquet (overwrite the same outputs the dashboard reads)
        with timer("aggregations"):
            setl.write_pandas_parquet(setl.kpi_daily(df),    config.KPI_DAILY_PARQUET)
            setl.write_pandas_parquet(setl.kpi_country(df),  config.KPI_COUNTRY_PARQUET)
            setl.write_pandas_parquet(setl.kpi_category(df), config.KPI_CATEGORY_PARQUET)
            setl.write_pandas_parquet(setl.kpi_brand(df),    config.KPI_BRAND_PARQUET)
            setl.write_pandas_parquet(setl.kpi_channel(df),  config.KPI_CHANNEL_PARQUET)

        # 4. Spark MLlib (Part A)
        if not args.skip_models:
            train_df = df
            if args.quick:
                train_df = df.sample(fraction=0.1, seed=42)
            with timer("Spark MLlib classifier"):
                clf = train_spark_classifier(train_df, target="is_returned")
            with timer("Spark MLlib regressor"):
                reg = train_spark_regressor(train_df, target="profit_usd")
            log.info("Spark MLlib classifier metrics: %s", clf.metrics)
            log.info("Spark MLlib regressor metrics:  %s", reg.metrics)

        # 5. Recommender (Part B) — collect to pandas for fitting (still big-data ALS via Spark)
        if not args.skip_recsys:
            with timer("recsys: collect to pandas"):
                pdf = df.select(
                    "customer_id", "product_id", "product_name", "category",
                    "sub_category", "brand", "order_id", "total_price_usd",
                    "rating", "unit_price_usd", "product_reviews_count",
                ).toPandas()

            with timer("recsys: build interactions + catalog"):
                interactions = rec.build_interactions(pdf)
                catalog = rec.build_product_catalog(pdf)

            rec.REC_DIR.mkdir(parents=True, exist_ok=True)
            interactions.to_parquet(rec.INTERACTIONS_PATH, index=False)

            with timer("recsys: train ALS"):
                als = rec.train_als(spark, interactions)
                log.info("ALS RMSE: %.4f", als.rmse)

            with timer("recsys: train content"):
                rec.train_content(catalog)

    spark.stop()
    log.info("Spark pipeline complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
