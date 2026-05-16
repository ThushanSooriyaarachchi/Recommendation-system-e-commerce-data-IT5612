"""End-to-end pipeline: ingest CSV -> clean parquet -> KPIs -> models.

Usage (from the project root, with the venv activated):

    python run_pipeline.py [--skip-models] [--quick]
"""

from __future__ import annotations

import argparse
import sys

import pandas as pd

from src import config
from src.data_loader import load_raw, downcast_numeric
from src.features import (
    compute_rfm,
    kpi_brand,
    kpi_category,
    kpi_channel,
    kpi_country,
    kpi_daily,
)
from src.models import (
    save,
    train_classifier,
    train_clustering,
    train_forecaster,
    train_regressor,
)
from src.preprocessing import clean
from src.utils import get_logger, timer

log = get_logger("run_pipeline")


def build_clean(force: bool = False) -> pd.DataFrame:
    if config.CLEAN_PARQUET.exists() and not force:
        log.info("Reusing existing clean parquet at %s", config.CLEAN_PARQUET)
        return pd.read_parquet(config.CLEAN_PARQUET)

    raw = load_raw()
    df = clean(raw)
    df = downcast_numeric(df)
    config.INTERIM_DIR.mkdir(parents=True, exist_ok=True)
    df.to_parquet(config.CLEAN_PARQUET, index=False)
    log.info("Wrote %s (%s rows)", config.CLEAN_PARQUET, f"{len(df):,}")
    return df


def build_kpis(df: pd.DataFrame) -> None:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    pairs = {
        config.KPI_DAILY_PARQUET: kpi_daily(df),
        config.KPI_COUNTRY_PARQUET: kpi_country(df),
        config.KPI_CATEGORY_PARQUET: kpi_category(df),
        config.KPI_BRAND_PARQUET: kpi_brand(df),
        config.KPI_CHANNEL_PARQUET: kpi_channel(df),
    }
    for path, frame in pairs.items():
        frame.to_parquet(path, index=False)
        log.info("Wrote %s (%s rows)", path.name, f"{len(frame):,}")


def build_rfm(df: pd.DataFrame) -> pd.DataFrame:
    rfm = compute_rfm(df)
    rfm.to_parquet(config.RFM_PARQUET, index=False)
    log.info("Wrote %s (%s customers)", config.RFM_PARQUET.name, f"{len(rfm):,}")
    return rfm


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df.to_parquet(config.FEATURES_PARQUET, index=False)
    log.info("Wrote %s", config.FEATURES_PARQUET.name)
    return df


def maybe_subsample(df: pd.DataFrame, quick: bool) -> pd.DataFrame:
    n = config.TRAIN_SAMPLE_ROWS
    if quick:
        n = min(50_000, len(df))
    if n is not None and len(df) > n:
        log.info("Subsampling to %s rows for training", f"{n:,}")
        return df.sample(n=n, random_state=config.RANDOM_STATE).reset_index(drop=True)
    return df


def train_all(df: pd.DataFrame, daily: pd.DataFrame, rfm: pd.DataFrame, quick: bool) -> None:
    sample = maybe_subsample(df, quick=quick)

    with timer("train classifier"):
        clf = train_classifier(sample, target="is_returned")
        save(clf, config.CLF_MODEL_PATH)

    with timer("train regressor"):
        reg = train_regressor(sample, target="profit_usd")
        save(reg, config.REG_MODEL_PATH)

    with timer("train clustering"):
        cluster = train_clustering(rfm)
        save(cluster, config.CLUSTER_MODEL_PATH)

    with timer("train forecaster"):
        fc = train_forecaster(daily, value_col="revenue")
        save(fc, config.FORECAST_MODEL_PATH)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-models", action="store_true", help="Only build parquet files.")
    parser.add_argument("--force-clean", action="store_true", help="Rebuild clean parquet even if it exists.")
    parser.add_argument("--quick", action="store_true", help="Use a 50k sample for model training.")
    args = parser.parse_args()

    config.ensure_dirs()

    with timer("Pipeline"):
        df = build_clean(force=args.force_clean)
        build_kpis(df)
        rfm = build_rfm(df)
        build_features(df)

        if args.skip_models:
            log.info("Skipping model training as requested.")
            return 0

        daily = kpi_daily(df)
        train_all(df, daily, rfm, quick=args.quick)

    log.info("Pipeline complete. Now run: streamlit run dashboard/app.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
