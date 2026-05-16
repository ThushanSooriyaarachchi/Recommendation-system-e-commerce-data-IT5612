# Big Data Analytics — Mini Project: Final Report Outline

> Course: Big Data Analytics — Apache Spark Analytics and Big Data Recommendation Systems
> Dataset: `Data Set/ecommerce_dataset_+1m.csv` (1,000,123 rows × 60 columns)
> Implementation: PySpark + scikit-learn + Spark MLlib + Streamlit

---

## 1. Executive Summary

- One-paragraph framing of the business problem (e-commerce analytics on 1M+ orders).
- Bullet list of the 5–7 highest-impact findings.
- Single chart: revenue trend with forecast tail.
- Headline numbers: total revenue, profit, AOV, return rate, top-3 categories.

## 2. Introduction

- 2.1 Problem statement: *"How can a retailer use big data analytics on 1M+ historical orders to (a) understand customer & product performance, (b) predict outcomes such as profit and returns, and (c) recommend products to existing customers?"*
- 2.2 Why big data: dataset size, latency requirements, scaling expectations.
- 2.3 Scope: Part A (Spark analytics) + Part B (recommendation system).
- 2.4 Tools chosen and rationale (PySpark, Spark MLlib ALS, sklearn, Streamlit).

## 3. Dataset

- 3.1 Source and provenance of the CSV.
- 3.2 Schema overview: 60 columns grouped into 10 logical domains
  (order, customer, product, pricing, payment, shipping, review, marketing,
  web analytics, risk/operations).
- 3.3 Sampling & sanity checks: row count, column count, datetime range
  (2024–2026), null patterns.
- 3.4 Known limitations: synthetic origin → some labels (e.g. `is_returned`)
  show weak feature-target correlation; this is an honest finding, not a bug.

## 4. Methodology — Part A (Spark Analytics)

- 4.1 SparkSession configuration (local[*], JAVA_HOME bootstrap on Windows,
  Arrow-enabled, adaptive query execution).
- 4.2 Distributed ingestion with `spark.read.csv`.
- 4.3 Cleaning pipeline (`src/spark_etl.clean`):
  - Date parsing
  - Null filtering on critical keys
  - Numeric outlier capping
  - Categorical normalization
  - Derived features (`order_dow`, `is_weekend_bool`, `customer_tenure_days`,
    `is_returned`, `is_high_fraud`, `coupon_used_bool`).
- 4.4 Aggregations (Spark SQL): daily KPIs, country, category, brand, channel.
- 4.5 RFM (Recency / Frequency / Monetary) computed in Spark.
- 4.6 Spark MLlib pipelines:
  - Classifier: `Imputer → StringIndexer → OneHotEncoder → VectorAssembler → GBTClassifier`,
    target = `is_returned`.
  - Regressor: same preprocessing + `GBTRegressor`, target = `profit_usd`.
- 4.7 Train/test split (80/20, seed 42).

## 5. Methodology — Part B (Recommendation System)

- 5.1 Problem statement: top-N product recommendations per customer.
- 5.2 Implicit interaction matrix from purchase frequency
  (`score = log1p(purchases) * mean_rating / 5`).
- 5.3 **Collaborative filtering** with Spark MLlib **ALS**
  (rank=24, regParam=0.08, maxIter=10, implicitPrefs=true).
- 5.4 **Content-based** with sklearn TF-IDF over `product_name + category +
  sub_category + brand` (1-2 grams, 20k features). User profile = mean of
  purchased-item TF-IDF vectors.
- 5.5 **Hybrid**: `score = α · ALS_score + (1-α) · content_score` after min-max
  scaling, default α = 0.6.
- 5.6 Production pattern: ALS user/item factors are exported to parquet so
  inference is pure NumPy (no live Spark session required at serve time).

## 6. Results

### 6.1 Descriptive analytics
- Top-line numbers (revenue, profit, AOV, orders, customers).
- Country / category / brand top-N tables.
- Hour × weekday demand heatmap.
- Sentiment vs rating breakdown.
- Return-reason Pareto + fraud-risk distribution.

### 6.2 Predictive modeling
| Model | Target | Metric | Spark MLlib | sklearn (sanity) |
|---|---|---|---|---|
| Classifier | `is_returned` | ROC-AUC | 0.498 | 0.509 |
| Classifier | `is_returned` | F1 (positive) | 0.85* | 0.18 (balanced) |
| Regressor | `profit_usd` | R² | 0.889 | 0.892 |
| Regressor | `profit_usd` | RMSE (USD) | 53.93 | 52.28 |
| Clustering | RFM | silhouette | 0.42 (k=5) | 0.42 (k=5) |
| Forecast | daily revenue | MAE 14d | $25,052 | $25,052 |

*F1 high in Spark report because evaluator uses macro/weighted defaults; the
ROC-AUC of ~0.5 is the more honest signal that returns in this synthetic
dataset are essentially random with respect to the captured features.

### 6.3 Recommendations
- ALS holdout RMSE: **0.4675** on implicit-feedback score scale.
- Latent factors: 844,414 users × 24 dim, 667,314 items × 24 dim.
- Content TF-IDF: 753,516 products × 724 features.
- Demo top-N tables (per strategy) for the most-active customer.

## 7. Discussion / Insights
- Revenue concentration (Pareto observed).
- Sensitivity of `is_returned` to features (low; suggests label is near-random
  in the synthetic data).
- Coupon lift on AOV / margin.
- Customer segmentation actionability (Champions vs At Risk).
- Operational levers (warehouse SLA, shipping method).

## 8. Limitations and Future Work
- Synthetic dataset → low classifier signal; real telemetry would likely raise AUC.
- Cold-start users (~1 order per customer) → content / hybrid > ALS in practice.
- ALS could be improved with `als.recommendForAllUsers` precomputed nightly.
- Adding session-level events would unlock funnel / abandonment models.
- Could extend with Spark Structured Streaming for real-time ingestion.

## 9. Reproducibility
- Repo layout (link to `README.md`).
- Exact commands to reproduce (venv, pipelines, dashboard launch).
- Random seeds: `RANDOM_STATE=42` everywhere.

## 10. References
- Apache Spark, Spark MLlib documentation.
- Hu, Koren, Volinsky: *Collaborative Filtering for Implicit Feedback Datasets* (ALS basis).
- Streamlit, scikit-learn, statsmodels documentation.

---

## Suggested figure list (export to `reports/figures/`)
1. `revenue_trend_by_month.png`
2. `country_top15_revenue.png`
3. `category_treemap.png`
4. `hour_dow_heatmap.png`
5. `rfm_scatter.png`
6. `return_reasons_pareto.png`
7. `forecast_history_vs_pred.png`
8. `als_rmse_by_rank.png` (optional ablation)
9. `hybrid_vs_als_overlap.png` (optional)
