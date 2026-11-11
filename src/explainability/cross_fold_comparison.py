
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_ROOT = Path.cwd()

BASE_DIR = PROJECT_ROOT / "results" / "explainability"

FOLDS = [1, 2, 3]
TOP_K = 20

OUTPUT_DIR = BASE_DIR / "cross_fold_comparison"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# LOAD FOLD RESULTS
# ==================================================

def load_fold_results():

    fold_data = {}

    for fold in FOLDS:

        # Fold 1 has a different folder name
        if fold == 1:
            folder_name = "integrated_gradients"
        else:
            folder_name = f"fold{fold}_integrated_gradients"

        file_path = (
            BASE_DIR
            / folder_name
            / "feature_importance_overall.csv"
        )

        if not file_path.exists():
            raise FileNotFoundError(
                f"Missing Fold {fold} results: {file_path}"
            )

        df = pd.read_csv(file_path)

        required_columns = {
            "feature_index",
            "mean_absolute_attribution",
        }

        if not required_columns.issubset(df.columns):
            raise ValueError(
                f"Unexpected CSV columns in Fold {fold}: "
                f"{df.columns.tolist()}"
            )

        df = df.sort_values(
            "mean_absolute_attribution",
            ascending=False,
        ).reset_index(drop=True)

        # Normalize importance within each fold
        total_importance = df[
            "mean_absolute_attribution"
        ].sum()

        if total_importance <= 0:
            raise ValueError(
                f"Invalid total importance in Fold {fold}"
            )

        df["normalized_importance"] = (
            df["mean_absolute_attribution"]
            / total_importance
        )

        df["rank"] = np.arange(1, len(df) + 1)

        fold_data[fold] = df

        print(
            f"Fold {fold} loaded successfully: "
            f"{len(df)} features"
        )

    return fold_data


# ==================================================
# CROSS-FOLD COMPARISON
# ==================================================

def create_comparison(fold_data):

    comparison = None

    for fold, df in fold_data.items():

        temp = df[
            [
                "feature_index",
                "mean_absolute_attribution",
                "normalized_importance",
                "rank",
            ]
        ].copy()

        temp.columns = [
            "feature_index",
            f"fold{fold}_importance",
            f"fold{fold}_normalized",
            f"fold{fold}_rank",
        ]

        if comparison is None:
            comparison = temp
        else:
            comparison = comparison.merge(
                temp,
                on="feature_index",
                how="inner",
            )

    normalized_columns = [
        f"fold{fold}_normalized"
        for fold in FOLDS
    ]

    rank_columns = [
        f"fold{fold}_rank"
        for fold in FOLDS
    ]

    # Equal-weight average of normalized importance
    comparison["mean_normalized_importance"] = (
        comparison[normalized_columns].mean(axis=1)
    )

    comparison["mean_rank"] = (
        comparison[rank_columns].mean(axis=1)
    )

    comparison = comparison.sort_values(
        "mean_normalized_importance",
        ascending=False,
    ).reset_index(drop=True)

    comparison.to_csv(
        OUTPUT_DIR / "cross_fold_feature_comparison.csv",
        index=False,
    )

    return comparison


# ==================================================
# TOP FEATURE OVERLAP
# ==================================================

def analyze_top_features(fold_data):

    top_features = {}

    for fold, df in fold_data.items():

        top_features[fold] = set(
            df.head(TOP_K)["feature_index"].astype(int)
        )

    common_features = (
        top_features[1]
        & top_features[2]
        & top_features[3]
    )

    print("\n" + "=" * 60)
    print("CROSS-FOLD TOP FEATURE ANALYSIS")
    print("=" * 60)

    for fold in FOLDS:

        print(f"\nFold {fold} Top {TOP_K} Features:")

        print(
            sorted(top_features[fold])
        )

    print("\nCommon Top 20 Features Across All Folds:")

    print(sorted(common_features))

    print(
        "\nNumber of common features:",
        len(common_features),
    )

    # Save common feature list
    pd.DataFrame({
        "common_feature_index": sorted(common_features)
    }).to_csv(
        OUTPUT_DIR / "common_top20_features.csv",
        index=False,
    )

    # Pairwise Jaccard similarity of top-20 sets
    pairwise_results = []

    for i in range(len(FOLDS)):
        for j in range(i + 1, len(FOLDS)):

            fold_a = FOLDS[i]
            fold_b = FOLDS[j]

            intersection = (
                top_features[fold_a]
                & top_features[fold_b]
            )

            union = (
                top_features[fold_a]
                | top_features[fold_b]
            )

            jaccard = (
                len(intersection) / len(union)
                if union else 0.0
            )

            pairwise_results.append({
                "fold_a": fold_a,
                "fold_b": fold_b,
                "common_features": len(intersection),
                "jaccard_similarity": jaccard,
            })

    pairwise_df = pd.DataFrame(pairwise_results)

    pairwise_df.to_csv(
        OUTPUT_DIR / "top20_overlap.csv",
        index=False,
    )

    print("\nPairwise Top-20 Overlap:")

    print(pairwise_df.to_string(index=False))

    return common_features


# ==================================================
# VISUALIZATION
# ==================================================

def create_visualization(comparison):

    top20 = comparison.head(TOP_K).copy()

    # Sort for horizontal bar chart
    top20 = top20.sort_values(
        "mean_normalized_importance",
        ascending=True,
    )

    plt.figure(figsize=(10, 8))

    plt.barh(
        top20["feature_index"].astype(str),
        top20["mean_normalized_importance"],
    )

    plt.xlabel("Mean Normalized Importance")
    plt.ylabel("Feature Index")

    plt.title(
        "Top 20 FI-2010 Features Across Three Folds\n"
        "Integrated Gradients"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "cross_fold_top20.png",
        dpi=300,
    )

    plt.close()

    # Individual fold comparison for consensus top 20
    selected = comparison.head(TOP_K).copy()

    x = np.arange(len(selected))
    width = 0.25

    plt.figure(figsize=(14, 7))

    for index, fold in enumerate(FOLDS):

        plt.bar(
            x + (index - 1) * width,
            selected[f"fold{fold}_normalized"],
            width,
            label=f"Fold {fold}",
        )

    plt.xticks(
        x,
        selected["feature_index"].astype(str),
        rotation=45,
    )

    plt.xlabel("Feature Index")
    plt.ylabel("Normalized Importance")

    plt.title(
        "Feature Importance Across Folds"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "fold_wise_feature_comparison.png",
        dpi=300,
    )

    plt.close()


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 60)
    print("CROSS-FOLD INTEGRATED GRADIENTS COMPARISON")
    print("=" * 60)

    # Load all three folds
    fold_data = load_fold_results()

    # Compare feature importance
    comparison = create_comparison(fold_data)

    # Analyze common features and overlap
    common_features = analyze_top_features(fold_data)

    # Generate graphs
    create_visualization(comparison)

    # Print consensus ranking
    print("\n" + "=" * 60)
    print("OVERALL TOP 20 CONSENSUS FEATURES")
    print("=" * 60)

    print(
        comparison.head(TOP_K)[
            [
                "feature_index",
                "mean_normalized_importance",
                "mean_rank",
            ]
        ].to_string(index=False)
    )

    print("\nCross-fold comparison completed.")

    print("Common features:", len(common_features))

    print("Results saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
