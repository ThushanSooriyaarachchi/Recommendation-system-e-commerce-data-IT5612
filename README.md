# Big Data Analytics — E-Commerce Mini Project

End-to-end big-data analytics solution built on a **1,000,123-row** e-commerce
dataset, implementing the full assignment rubric:

- **Part A — Big Data Analytics with Apache Spark / PySpark**
  - Distributed ETL, Spark SQL aggregations, Spark MLlib classifier + regressor.
- **Part B — Recommendation System**
  - Spark MLlib **ALS** (collaborative filtering) +
    **content-based** TF-IDF + **hybrid** weighted blend.
- **Visualisation layer** — multi-page **Streamlit** dashboard
  (12 themed pages, including a live Recommendations page).

## Project layout

```
Assignment/
├── Data Set/                       # raw CSV (do not modify)
├── Assignment guideline/           # rubric PDF
├── data/
│   ├── interim/clean.parquet       # cleaned dataset
│   └── processed/*.parquet         # KPI / RFM / features
├── notebooks/
│   └── 01_spark_eda_and_modelling.ipynb
├── src/
│   ├── config.py
│   ├── data_loader.py              # pandas chunked loader
│   ├── preprocessing.py            # pandas cleaner
│   ├── features.py                 # RFM, KPIs, feature lists
│   ├── models.py                   # sklearn classifier / regressor / clustering / forecaster
│   ├── evaluation.py
│   ├── utils.py
│   ├── spark_session.py            # Windows-safe SparkSession factory
│   ├── spark_etl.py                # PySpark ETL + aggregations
│   ├── spark_models.py             # Spark MLlib classifier + regressor
│   └── recommender.py              # ALS + content-based + hybrid recommenders
├── dashboard/
│   ├── app.py                      # Home + global filters
│   ├── pages/                      # 1..12 themed Streamlit pages
│   ├── components/                 # filters, KPI cards, chart helpers
│   ├── theme/style.css
│   └── .streamlit/config.toml
├── models/
│   ├── classifier.joblib           # sklearn HGBT
│   ├── regressor.joblib            # sklearn GBT
│   ├── clustering.joblib           # KMeans on RFM
│   ├── forecaster.joblib           # Holt-Winters
│   └── recommender/
│       ├── als_user_factors.parquet
│       ├── als_item_factors.parquet
│       ├── als_meta.json
│       ├── user_index.parquet
│       ├── item_index.parquet
│       ├── product_catalog.parquet
│       ├── interactions.parquet
│       └── content.joblib
├── reports/
│   ├── figures/
│   └── final_report_outline.md
├── tests/test_preprocessing.py
├── scripts/                        # one-off helpers (retrain / smoke test)
├── run_pipeline.py                 # pandas + sklearn pipeline
├── run_spark_pipeline.py           # PySpark + Spark MLlib + recommender pipeline
├── requirements.txt
├── .gitignore
└── README.md
```

## Prerequisites

- Windows 10/11 PowerShell
- **Python 3.12** (`py -3.12 --version`)
- **Java 17 LTS** (PySpark requires it). Tested with `C:\Program Files\Java\jdk-17`.
  `JAVA_HOME` is auto-detected by `src/spark_session.py` if it isn't set.

## Quick start

```powershell
cd "d:\MSC ACA\SEM 3\Assignment"

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python -m ipykernel install --user --name=ecom-assignment --display-name "Python (ecom-assignment)"
```

## Build everything

There are **two** pipelines. They write to the **same** parquet outputs that the
dashboard reads, so the dashboard works after either one.

### Option 1 — Spark pipeline (Part A + Part B, the rubric path)

```powershell
python run_spark_pipeline.py            # full ETL + Spark MLlib + ALS + content
# or
python run_spark_pipeline.py --quick    # uses 10% sample for MLlib training
```

What it does:
1. Spark reads the CSV (1,000,123 rows).
2. Spark SQL cleaning + feature engineering.
3. Writes pre-aggregated parquet for the dashboard.
4. Trains Spark MLlib classifier (`is_returned`) and regressor (`profit_usd`).
5. Trains Spark MLlib **ALS** on the implicit-feedback purchase matrix.
6. Exports ALS user/item latent factors to parquet for fast inference.
7. Trains the **content-based** TF-IDF recommender.

### Option 2 — pandas + sklearn pipeline (laptop-friendly fallback)

```powershell
python run_pipeline.py
```

This rebuilds the same parquet outputs and trains sklearn versions of the
classifier (HistGradientBoosting), regressor (GradientBoosting), clustering
(KMeans) and forecaster (Holt-Winters).

## Launch the dashboard

```powershell
streamlit run dashboard/app.py
```

The Streamlit app is multi-page. Use the sidebar global filters (date, country,
category, segment) which propagate via `st.session_state` to every page.

| Page | Theme |
|---|---|
| Home | Headline KPIs + global filters |
| Overview KPIs | Revenue, profit, MoM growth, status mix |
| Sales Trends | Daily / weekly / monthly + hour × weekday heatmap |
| Geography | Country choropleth + top cities |
| Products & Brands | Treemap + brand scatter + top products |
| Customers RFM | Segment pie/bar + RFM scatter |
| Marketing & Channels | Coupon lift, traffic source, device |
| Operations | Shipping method, delivery SLA, warehouses |
| Reviews & Sentiment | Rating dist, sentiment mix, wordcloud |
| Returns & Fraud | Return-reason Pareto + fraud-risk hist |
| Forecasting | Holt-Winters daily revenue forecast |
| Predict Live | Score a synthetic order against trained models |
| **Recommendations** | **ALS + content + hybrid (Part B)** |

## Running the notebook

```powershell
jupyter lab
# open notebooks/01_spark_eda_and_modelling.ipynb
# select kernel: "Python (ecom-assignment)"
```

## Tests

```powershell
pytest -q
```

## Performance notes

- The 1M-row CSV takes ~13 s to load in pandas, ~7 s in Spark on a laptop.
- All KPI parquet files are pre-aggregated, so dashboard pages load in <1 s.
- ALS factors are stored as parquet so recommendation inference is pure NumPy
  (no live Spark session required when the dashboard runs).
- Sidebar offers a **dataset size toggle** (100k / 250k / full) so reviewers
  can interact fast and still validate against the full dataset.

## Reproducibility

- All randomness is seeded via `src/config.RANDOM_STATE = 42`.
- Pipeline outputs are deterministic given the same raw CSV.
- The Streamlit app loads only deterministic parquet artefacts.
