
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

from src.data.fi2010 import FI2010Loader, HORIZONS

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

    for split in ["Training", "Testing"]:

        X, y = loader.load_fold(fold, split=split)

        for h, horizon in enumerate(HORIZONS):

            labels, counts = np.unique(
                y[:, h],
                return_counts=True
            )

            for label, count in zip(labels, counts):

                rows.append({
                    "fold": fold,
                    "split": split,
                    "horizon": horizon,
                    "raw_label": int(label),
                    "label_name": {
                        1: "Up",
                        2: "Stationary",
                        3: "Down"
                    }[int(label)],
                    "count": int(count),
                    "percentage": 100 * count / len(y)
                })

df = pd.DataFrame(rows)

output_path = OUTPUT / "label_structure_audit.csv"
df.to_csv(output_path, index=False)

print("\nLABEL STRUCTURE AUDIT")
print("=" * 65)

for split in ["Training", "Testing"]:

    print(f"\n{split.upper()}")

    summary = df[df["split"] == split].groupby(
        ["horizon", "label_name"]
    )["percentage"].mean().unstack()

    print(summary.round(2).to_string())

print("\nRaw label values:", sorted(df["raw_label"].unique()))

print("\nSaved:", output_path)
