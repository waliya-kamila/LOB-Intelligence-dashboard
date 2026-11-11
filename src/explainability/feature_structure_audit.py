
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.data.fi2010 import FI2010Loader


ROOT = (
    Path.home()
    / "Downloads"
    / "BenchmarkDatasets"
    / "NoAuction"
    / "1.NoAuction_Zscore"
)

OUTPUT = Path("results/explainability")
OUTPUT.mkdir(parents=True, exist_ok=True)

loader = FI2010Loader(str(ROOT))

rows = []

for fold in [1, 2, 3]:

    X, y = loader.load_fold(fold, split="Training")

    for feature_idx in range(X.shape[1]):

        values = X[:, feature_idx]

        rows.append({
            "fold": fold,
            "feature_index": feature_idx,
            "dataset_row": feature_idx + 1,
            "mean": values.mean(),
            "std": values.std(),
            "min": values.min(),
            "max": values.max(),
            "unique_values": np.unique(values).size,
            "is_constant": values.std() == 0
        })

df = pd.DataFrame(rows)

path = OUTPUT / "feature_structure_audit.csv"
df.to_csv(path, index=False)

summary = df.groupby("feature_index").agg(
    mean_std=("std", "mean"),
    min_value=("min", "min"),
    max_value=("max", "max"),
    constant_folds=("is_constant", "sum")
).reset_index()

print("\nFEATURE STRUCTURE AUDIT")
print("=" * 60)

print("Total features:", df["feature_index"].nunique())

print(
    "Features constant in all three folds:",
    (summary["constant_folds"] == 3).sum()
)

print("\nFeatures with zero standard deviation:")
print(
    summary.loc[
        summary["constant_folds"] == 3,
        ["feature_index", "mean_std"]
    ].to_string(index=False)
)

print("\nFirst 15 feature statistics:")
print(summary.head(15).round(5).to_string(index=False))

print("\nSaved:", path)
