
import pandas as pd
from pathlib import Path

folders = {
    1: Path("results/explainability/integrated_gradients"),
    2: Path("results/explainability/fold2_integrated_gradients"),
    3: Path("results/explainability/fold3_integrated_gradients"),
}

class_names = {
    0: "Up",
    1: "Stationary",
    2: "Down",
}

groups = {
    "Basic LOB": (0, 40),
    "Time-insensitive": (40, 86),
    "Time-sensitive": (86, 144),
}

results = []

for fold, folder in folders.items():
    path = folder / "feature_importance_by_class.csv"

    if not path.exists():
        raise FileNotFoundError(path)

    df = pd.read_csv(path)

    for class_id, class_name in class_names.items():
        class_df = df[df["class_id"] == class_id]
        total = class_df["mean_absolute_attribution"].sum()

        if len(class_df) != 144 or total <= 0:
            raise ValueError(
                f"Unexpected data for fold {fold}, class {class_name}"
            )

        for group, (start, end) in groups.items():
            values = class_df[
                class_df["feature_index"].between(start, end - 1)
            ]["mean_absolute_attribution"]

            attribution = values.sum()

            results.append({
                "fold": fold,
                "class": class_name,
                "feature_group": group,
                "attribution": attribution,
                "percentage": 100 * attribution / total,
            })

result_df = pd.DataFrame(results)

print("\nCLASS-WISE FEATURE-FAMILY ATTRIBUTION")
print(result_df.to_string(index=False))

summary = result_df.groupby(
    ["class", "feature_group"]
).agg(
    mean_percentage=("percentage", "mean"),
    std_percentage=("percentage", "std")
).reset_index()

print("\nCROSS-FOLD CLASS-WISE SUMMARY")
print(summary.to_string(index=False))

output = Path(
    "results/explainability/classwise_feature_group_attribution.csv"
)

output.parent.mkdir(parents=True, exist_ok=True)
result_df.to_csv(output, index=False)

summary_output = Path(
    "results/explainability/classwise_feature_group_summary.csv"
)
summary.to_csv(summary_output, index=False)

print("\nSaved:", output)
print("Saved:", summary_output)

