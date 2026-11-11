
import pandas as pd
import numpy as np
from pathlib import Path

path = Path("results/multifold_results.csv")
df = pd.read_csv(path)

print("\nMULTIFOLD RESULTS")
print("=" * 65)
print(df.to_string(index=False))

# Correct column name: model, not model_name
print("\nMODEL NAMES:")
print(df["model"].unique())

print("\nFOLD COUNTS:")
print(df.groupby("model")["fold"].nunique())

# Pivot into paired fold-wise comparisons
pivot = df.pivot(
    index="fold",
    columns="model",
    values="test_macro_f1"
)

no_attention = pivot["CNN-GRU-NoAttention"]
attention = pivot["CNN-GRU-Attention"]

comparison = pd.DataFrame({
    "NoAttention": no_attention,
    "Attention": attention,
    "Difference_Attention_minus_NoAttention":
        attention - no_attention
})

print("\nPAIRED FOLD COMPARISON")
print(comparison.to_string())

print("\nSUMMARY")
for name, values in [
    ("NoAttention", no_attention),
    ("Attention", attention),
]:
    print(
        f"{name}: Mean={values.mean():.6f}, "
        f"SD={values.std(ddof=1):.6f}"
    )

differences = attention - no_attention

print("\nPAIRED DIFFERENCE SUMMARY")
print("Mean difference:", differences.mean())
print("SD of differences:", differences.std(ddof=1))
print("Attention wins:", int((differences > 0).sum()))
print("NoAttention wins:", int((differences < 0).sum()))
print("Ties:", int((differences == 0).sum()))

output = Path(
    "results/evaluation_multifold/statistical_comparison.csv"
)
output.parent.mkdir(parents=True, exist_ok=True)
comparison.to_csv(output)

print("\nSaved:", output)
