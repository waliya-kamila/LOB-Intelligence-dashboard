
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

INPUT = Path(
    "results/evaluation_multifold/"
    "horizon10_prediction_probabilities.csv"
)

OUTPUT = Path("results/evaluation_multifold")
df = pd.read_csv(INPUT)

CLASS_NAMES = ["Up", "Stationary", "Down"]
PROB_COLS = ["prob_up", "prob_stationary", "prob_down"]
N_BINS = 15


def calculate_metrics(group):

    probabilities = group[PROB_COLS].to_numpy()
    actual = group["actual_class"].map(
        {name: i for i, name in enumerate(CLASS_NAMES)}
    ).to_numpy()

    predicted = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    correct = (predicted == actual).astype(float)

    # Expected Calibration Error
    ece = 0.0
    bin_rows = []

    bin_edges = np.linspace(0, 1, N_BINS + 1)

    for i in range(N_BINS):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == N_BINS - 1:
            mask = (confidence >= lower) & (confidence <= upper)
        else:
            mask = (confidence >= lower) & (confidence < upper)

        count = mask.sum()

        if count == 0:
            continue

        bin_confidence = confidence[mask].mean()
        bin_accuracy = correct[mask].mean()

        ece += (
            count / len(group)
        ) * abs(bin_accuracy - bin_confidence)

        bin_rows.append({
            "model": group["model"].iloc[0],
            "bin": i + 1,
            "lower_bound": lower,
            "upper_bound": upper,
            "sample_count": count,
            "mean_confidence": bin_confidence,
            "accuracy": bin_accuracy
        })

    # Multiclass Brier Score
    one_hot = np.eye(3)[actual]

    brier = np.mean(
        np.sum((probabilities - one_hot) ** 2, axis=1)
    )

    # Multiclass Negative Log-Likelihood
    true_probabilities = probabilities[
        np.arange(len(actual)), actual
    ]

    nll = -np.mean(
        np.log(np.clip(true_probabilities, 1e-12, 1.0))
    )

    return {
        "model": group["model"].iloc[0],
        "ECE": ece,
        "Brier_score": brier,
        "NLL": nll,
        "accuracy": correct.mean(),
        "mean_confidence": confidence.mean()
    }, bin_rows


summary_rows = []
all_bins = []

for model, group in df.groupby("model"):

    metrics, bins = calculate_metrics(group)

    summary_rows.append(metrics)
    all_bins.extend(bins)

summary = pd.DataFrame(summary_rows)
bins_df = pd.DataFrame(all_bins)

summary_path = OUTPUT / "horizon10_calibration_summary.csv"
bins_path = OUTPUT / "horizon10_reliability_bins.csv"

summary.to_csv(summary_path, index=False)
bins_df.to_csv(bins_path, index=False)

print("\nCALIBRATION COMPARISON")
print(summary.round(5).to_string(index=False))

print("\nRELIABILITY BIN DATA")
print(bins_df.round(4).to_string(index=False))

print("\nSaved:")
print(summary_path)
print(bins_path)

