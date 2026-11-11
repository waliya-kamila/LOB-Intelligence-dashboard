
import pandas as pd
from pathlib import Path

path = Path("results/evaluation_multifold/detailed_metrics.csv")
df = pd.read_csv(path)

# Keep only aggregate metric rows, not individual class rows
df = df[df["class"].isna()].copy()

pivot = df.pivot(
    index=["fold", "horizon"],
    columns="model",
    values="macro_f1"
).reset_index()

pivot["difference"] = (
    pivot["CNN-GRU-Attention"]
    - pivot["CNN-GRU-NoAttention"]
)

print("\nFOLD-WISE HORIZON COMPARISON")
print(pivot.to_string(index=False))

summary = pivot.groupby("horizon").agg(
    mean_difference=("difference", "mean"),
    std_difference=("difference", "std"),
    attention_wins=("difference", lambda x: (x > 0).sum()),
    no_attention_wins=("difference", lambda x: (x < 0).sum())
).reset_index()

print("\nHORIZON-WISE SUMMARY")
print(summary.to_string(index=False))

output = Path(
    "results/evaluation_multifold/horizon_comparison.csv"
)
pivot.to_csv(output, index=False)

print("\nSaved:", output)

