"""Project-wide paths and constants.

Importing this module from any script gives consistent absolute paths
regardless of the current working directory, which is critical for
notebooks vs Streamlit vs CLI runs.
"""

from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parents[1]

RAW_DIR: Path = PROJECT_ROOT / "Data Set"
RAW_CSV: Path = RAW_DIR / "ecommerce_dataset_+1m.csv"

DATA_DIR: Path = PROJECT_ROOT / "data"
INTERIM_DIR: Path = DATA_DIR / "interim"
PROCESSED_DIR: Path = DATA_DIR / "processed"

MODELS_DIR: Path = PROJECT_ROOT / "models"
REPORTS_DIR: Path = PROJECT_ROOT / "reports"
FIGURES_DIR: Path = REPORTS_DIR / "figures"

CLEAN_PARQUET: Path = INTERIM_DIR / "clean.parquet"
FEATURES_PARQUET: Path = PROCESSED_DIR / "features.parquet"
KPI_DAILY_PARQUET: Path = PROCESSED_DIR / "kpi_daily.parquet"
KPI_COUNTRY_PARQUET: Path = PROCESSED_DIR / "kpi_country.parquet"
KPI_CATEGORY_PARQUET: Path = PROCESSED_DIR / "kpi_category.parquet"
KPI_BRAND_PARQUET: Path = PROCESSED_DIR / "kpi_brand.parquet"
KPI_CHANNEL_PARQUET: Path = PROCESSED_DIR / "kpi_channel.parquet"
RFM_PARQUET: Path = PROCESSED_DIR / "rfm_segments.parquet"

CLF_MODEL_PATH: Path = MODELS_DIR / "classifier.joblib"
REG_MODEL_PATH: Path = MODELS_DIR / "regressor.joblib"
CLUSTER_MODEL_PATH: Path = MODELS_DIR / "clustering.joblib"
FORECAST_MODEL_PATH: Path = MODELS_DIR / "forecaster.joblib"

RANDOM_STATE: int = 42

# When the raw file is too big for development laptops we sample.
# Set to None for the full 1M+ rows; set an int to subsample the clean parquet
# only when training heavy models.
TRAIN_SAMPLE_ROWS: int | None = 200_000

CHUNK_SIZE: int = 100_000


def ensure_dirs() -> None:
    for d in (INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)
