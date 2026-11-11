
import pandas as pd
from pathlib import Path

path = Path("results/evaluation_multifold/detailed_metrics.csv")
df = pd.read_csv(path)

# Keep class-level rows
df = df[df["class"].notna()].copy()

# Compare class F1 across models
pivot = df.pivot(
    index=["fold", "horizon", "class"],
    columns="model",
    values="class_f1"
).reset_index()

pivot["difference"] = (
    pivot["CNN-GRU-Attention"]
    - pivot["CNN-GRU-NoAttention"]
)

print("\nCLASS-WISE HORIZON COMPARISON")
print(pivot.to_string(index=False))

summary = pivot.groupby(
    ["horizon", "class"]
).agg(
    mean_difference=("difference", "mean"),
    std_difference=("difference", "std"),
    attention_wins=("difference", lambda x: (x > 0).sum()),
    no_attention_wins=("difference", lambda x: (x < 0).sum())
).reset_index()

print("\nCROSS-FOLD SUMMARY")
print(summary.to_string(index=False))

output = Path(
    "results/evaluation_multifold/classwise_horizon_comparison.csv"
)
output.parent.mkdir(parents=True, exist_ok=True)
pivot.to_csv(output, index=False)

print("\nSaved:", output)
