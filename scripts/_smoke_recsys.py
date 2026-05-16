"""Smoke-test the three recommendation strategies."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import recommender as rec


def main() -> int:
    interactions = pd.read_parquet(rec.INTERACTIONS_PATH)
    catalog = pd.read_parquet(rec.PRODUCT_CATALOG_PATH)

    print(f"Interactions: {len(interactions):,}")
    print(f"Catalog:      {len(catalog):,} products")

    # Pick the most-active customer (the dataset is sparse: ~1 order/customer).
    counts = interactions.groupby("customer_id")["purchases"].sum().sort_values(ascending=False)
    customer = counts.index[0]
    print(f"\n=== Most-active customer: {customer} (purchases={int(counts.iloc[0])}) ===")

    print("\n--- ALS top-5 ---")
    print(rec.recommend_als(None, customer, top_n=5))

    print("\n--- Content-based for user, top-5 ---")
    print(rec.recommend_content_for_user(customer, interactions, top_n=5))

    pid = catalog.iloc[0]["product_id"]
    print(f"\n--- Content-based: similar to product {pid} (top-5) ---")
    print(rec.recommend_content_for_product(pid, top_n=5))

    print("\n--- Hybrid (alpha=0.6) top-5 ---")
    print(rec.recommend_hybrid(None, customer, interactions, top_n=5, alpha=0.6))

    return 0


if __name__ == "__main__":
    sys.exit(main())
