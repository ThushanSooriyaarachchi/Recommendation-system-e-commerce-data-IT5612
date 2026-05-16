"""One-off helper: retrain only the classifier from existing clean parquet."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd

from src import config
from src.models import save, train_classifier


def main() -> int:
    df = pd.read_parquet(config.CLEAN_PARQUET)
    print(f"Loaded {len(df):,} rows")

    n = config.TRAIN_SAMPLE_ROWS
    if n is not None and len(df) > n:
        sample = df.sample(n=n, random_state=config.RANDOM_STATE).reset_index(drop=True)
    else:
        sample = df

    print(f"Sampled to {len(sample):,} rows")
    print(f"Class balance is_returned: {sample['is_returned'].mean():.4f}")

    clf = train_classifier(sample, target="is_returned")
    save(clf, config.CLF_MODEL_PATH)

    print("Test set report:", clf.report.to_dict())
    return 0


if __name__ == "__main__":
    sys.exit(main())
