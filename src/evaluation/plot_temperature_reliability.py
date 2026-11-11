
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

INPUT = Path(
    "results/evaluation_multifold/"
    "temperature_scaling_predictions.csv"
)

OUTPUT = Path("results/evaluation_multifold")
df = pd.read_csv(INPUT)

N_BINS = 15

plt.figure(figsize=(10, 8))

plt.plot(
    [0, 1], [0, 1],
    linestyle="--",
    label="Perfect Calibration"
)

styles = [
    ("CNN-GRU-Attention", "Original", "o"),
    ("CNN-GRU-Attention", "Calibrated", "s"),
    ("CNN-GRU-NoAttention", "Original", "^"),
    ("CNN-GRU-NoAttention", "Calibrated", "D")
]

for model, stage, marker in styles:

    subset = df[
        (df["model"] == model) &
        (df["stage"] == stage)
    ]

    confidence = subset["confidence"].to_numpy()
    correct = subset["correct"].to_numpy()

    edges = np.linspace(0, 1, N_BINS + 1)

    bin_confidence = []
    bin_accuracy = []

    for i in range(N_BINS):

        mask = (
            (confidence >= edges[i]) &
            (confidence < edges[i + 1])
        )

        if i == N_BINS - 1:
            mask = (
                (confidence >= edges[i]) &
                (confidence <= edges[i + 1])
            )

        if mask.sum() == 0:
            continue

        bin_confidence.append(confidence[mask].mean())
        bin_accuracy.append(correct[mask].mean())

    label = f"{model} - {stage}"

    plt.plot(
        bin_confidence,
        bin_accuracy,
        marker=marker,
        linewidth=2,
        markersize=6,
        label=label
    )

plt.xlabel("Mean Predicted Confidence", fontsize=12)
plt.ylabel("Actual Accuracy", fontsize=12)

plt.title(
    "Reliability Diagram: Before vs After Temperature Scaling",
    fontsize=13
)

plt.xlim(0, 1)
plt.ylim(0, 1)

plt.grid(alpha=0.3)
plt.legend(fontsize=9)
plt.tight_layout()

output_path = OUTPUT / "temperature_scaling_reliability.png"

plt.savefig(output_path, dpi=300, bbox_inches="tight")
plt.show()

print("Saved:", output_path)
