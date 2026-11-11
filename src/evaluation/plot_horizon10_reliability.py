
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# Paths
INPUT = Path(
    "results/evaluation_multifold/"
    "horizon10_reliability_bins.csv"
)

OUTPUT_DIR = Path("results/evaluation_multifold")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)

models = [
    "CNN-GRU-Attention",
    "CNN-GRU-NoAttention"
]

# Plot
plt.figure(figsize=(9, 7))

# Perfect calibration reference
plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

for model in models:

    subset = df[df["model"] == model].sort_values("bin")

    plt.plot(
        subset["mean_confidence"],
        subset["accuracy"],
        marker="o",
        linewidth=2,
        markersize=6,
        label=model
    )

plt.xlabel("Mean Predicted Confidence", fontsize=12)
plt.ylabel("Actual Accuracy", fontsize=12)

plt.title(
    "Reliability Diagram - Horizon 10",
    fontsize=14
)

plt.xlim(0.30, 1.02)
plt.ylim(0.30, 1.02)

plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()

output_path = (
    OUTPUT_DIR / "horizon10_reliability_diagram.png"
)

plt.savefig(output_path, dpi=300, bbox_inches="tight")

plt.show()

print("\nReliability diagram saved:")
print(output_path)

