
import pandas as pd
from pathlib import Path

path = Path("results/evaluation_multifold/confusion_matrices.csv")
df = pd.read_csv(path)

# Focus on horizon 10
df = df[df["horizon"] == 10].copy()

classes = ["Up", "Stationary", "Down"]
models = ["CNN-GRU-NoAttention", "CNN-GRU-Attention"]

results = []

for fold in sorted(df["fold"].unique()):

    print(f"\n{'=' * 65}")
    print(f"FOLD {fold} - HORIZON 10")
    print("=" * 65)

    matrices = {}

    for model in models:

        subset = df[
            (df["fold"] == fold) &
            (df["model"] == model)
        ]

        matrix = subset.pivot(
            index="actual_class",
            columns="predicted_class",
            values="count"
        ).reindex(index=classes, columns=classes)

        matrices[model] = matrix

        # Row-normalized confusion matrix
        normalized = matrix.div(matrix.sum(axis=1), axis=0) * 100

        print(f"\n{model} - Row-normalized percentages")
        print(normalized.round(2).to_string())

        for actual in classes:
            total = matrix.loc[actual].sum()

            for predicted in classes:
                count = matrix.loc[actual, predicted]

                results.append({
                    "fold": fold,
                    "model": model,
                    "actual_class": actual,
                    "predicted_class": predicted,
                    "count": count,
                    "row_percentage": 100 * count / total
                })

    # Compare error percentages
    no_att = matrices["CNN-GRU-NoAttention"]
    att = matrices["CNN-GRU-Attention"]

    difference = (
        att.div(att.sum(axis=1), axis=0)
        - no_att.div(no_att.sum(axis=1), axis=0)
    ) * 100

    print("\nAttention minus No-Attention (percentage points)")
    print(difference.round(2).to_string())

results_df = pd.DataFrame(results)

output = Path(
    "results/evaluation_multifold/horizon10_confusion_analysis.csv"
)
output.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(output, index=False)

print("\nSaved:", output)

# Cross-fold mean row-normalized percentages
summary = results_df.groupby(
    ["model", "actual_class", "predicted_class"]
)["row_percentage"].mean().reset_index()

print("\nCROSS-FOLD MEAN CONFUSION PERCENTAGES")
print(summary.to_string(index=False))

summary.to_csv(
    "results/evaluation_multifold/horizon10_confusion_summary.csv",
    index=False
)

print("\nSummary saved.")
