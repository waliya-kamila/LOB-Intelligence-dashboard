
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

CLASS_NAMES = {
    0: "UP",
    1: "STATIONARY",
    2: "DOWN",
}

OUTPUT_DIR = BASE_DIR / "classwise_cross_fold"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# LOAD CLASS-WISE RESULTS
# ==================================================

def load_class_results(fold):

    if fold == 1:
        folder_name = "integrated_gradients"
    else:
        folder_name = f"fold{fold}_integrated_gradients"

    file_path = (
        BASE_DIR
        / folder_name
        / "feature_importance_by_class.csv"
    )

    if not file_path.exists():
        raise FileNotFoundError(
            f"Missing file for Fold {fold}: {file_path}"
        )

    df = pd.read_csv(file_path)

    required_columns = {
        "class_id",
        "feature_index",
        "mean_absolute_attribution",
    }

    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Unexpected columns in Fold {fold}: "
            f"{df.columns.tolist()}"
        )

    return df


# ==================================================
# NORMALIZE FEATURE IMPORTANCE
# ==================================================

def normalize_class_importance(df):

    df = df.copy()

    total = df["mean_absolute_attribution"].sum()

    if total <= 0:
        raise ValueError("Total attribution importance is invalid.")

    df["normalized_importance"] = (
        df["mean_absolute_attribution"] / total
    )

    df = df.sort_values(
        "mean_absolute_attribution",
        ascending=False,
    ).reset_index(drop=True)

    df["rank"] = np.arange(1, len(df) + 1)

    return df


# ==================================================
# PAIRWISE JACCARD SIMILARITY
# ==================================================

def calculate_jaccard(set_a, set_b):

    union = set_a | set_b

    if not union:
        return 0.0

    return len(set_a & set_b) / len(union)


# ==================================================
# CLASS-WISE CROSS-FOLD ANALYSIS
# ==================================================

def analyze_class(
    class_id,
    fold_data,
):

    class_name = CLASS_NAMES[class_id]

    print("\n" + "=" * 65)
    print(f"CLASS-WISE ANALYSIS: {class_name}")
    print("=" * 65)

    fold_results = {}
    top_feature_sets = {}

    # Prepare each fold's class-specific results
    for fold in FOLDS:

        df = fold_data[fold]

        class_df = df[
            df["class_id"] == class_id
        ].copy()

        if len(class_df) != 144:
            raise ValueError(
                f"Expected 144 features for {class_name}, "
                f"Fold {fold}; found {len(class_df)}"
            )

        class_df = normalize_class_importance(class_df)

        fold_results[fold] = class_df

        top_feature_sets[fold] = set(
            class_df.head(TOP_K)["feature_index"].astype(int)
        )

    # --------------------------------------------------
    # TOP FEATURES PER FOLD
    # --------------------------------------------------

    for fold in FOLDS:

        print(f"\nFold {fold} Top {TOP_K} Features:")

        print(
            sorted(top_feature_sets[fold])
        )

    # --------------------------------------------------
    # COMMON FEATURES
    # --------------------------------------------------

    common_features = (
        top_feature_sets[1]
        & top_feature_sets[2]
        & top_feature_sets[3]
    )

    print("\nCommon Top 20 Features Across All Folds:")

    print(sorted(common_features))

    print(
        "Number of common features:",
        len(common_features),
    )

    # --------------------------------------------------
    # PAIRWISE JACCARD
    # --------------------------------------------------

    pairwise_results = []

    for i in range(len(FOLDS)):

        for j in range(i + 1, len(FOLDS)):

            fold_a = FOLDS[i]
            fold_b = FOLDS[j]

            similarity = calculate_jaccard(
                top_feature_sets[fold_a],
                top_feature_sets[fold_b],
            )

            pairwise_results.append({
                "class": class_name,
                "fold_a": fold_a,
                "fold_b": fold_b,
                "common_features": len(
                    top_feature_sets[fold_a]
                    & top_feature_sets[fold_b]
                ),
                "jaccard_similarity": similarity,
            })

    pairwise_df = pd.DataFrame(pairwise_results)

    print("\nPairwise Jaccard Similarity:")

    print(
        pairwise_df.to_string(index=False)
    )

    # --------------------------------------------------
    # MERGE FEATURE IMPORTANCE ACROSS FOLDS
    # --------------------------------------------------

    comparison = None

    for fold, df in fold_results.items():

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

    comparison["class"] = class_name

    # --------------------------------------------------
    # SAVE CSV FILES
    # --------------------------------------------------

    comparison.to_csv(
        OUTPUT_DIR / f"{class_name.lower()}_feature_comparison.csv",
        index=False,
    )

    pd.DataFrame({
        "feature_index": sorted(common_features)
    }).to_csv(
        OUTPUT_DIR / f"{class_name.lower()}_common_features.csv",
        index=False,
    )

    pairwise_df.to_csv(
        OUTPUT_DIR / f"{class_name.lower()}_jaccard.csv",
        index=False,
    )

    # --------------------------------------------------
    # VISUALIZATION
    # --------------------------------------------------

    top20 = comparison.head(TOP_K).sort_values(
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
        f"{class_name}: Top 20 Features Across Folds"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / f"{class_name.lower()}_top20.png",
        dpi=300,
    )

    plt.close()

    return {
        "class": class_name,
        "common_features": sorted(common_features),
        "common_count": len(common_features),
        "mean_jaccard": pairwise_df[
            "jaccard_similarity"
        ].mean(),
        "comparison": comparison,
        "pairwise": pairwise_df,
    }


# ==================================================
# MAIN
# ==================================================

def main():

    print("=" * 65)
    print("CLASS-WISE CROSS-FOLD INTEGRATED GRADIENTS")
    print("=" * 65)

    # Load all folds
    fold_data = {}

    for fold in FOLDS:

        fold_data[fold] = load_class_results(fold)

        print(
            f"Fold {fold} loaded successfully."
        )

    all_results = []
    all_comparisons = []
    all_pairwise = []

    # Analyze UP, STATIONARY and DOWN separately
    for class_id in CLASS_NAMES:

        result = analyze_class(
            class_id,
            fold_data,
        )

        all_results.append({
            "class": result["class"],
            "common_top20_features": result["common_count"],
            "mean_jaccard_similarity": result["mean_jaccard"],
        })

        all_comparisons.append(
            result["comparison"]
        )

        all_pairwise.append(
            result["pairwise"]
        )

    # --------------------------------------------------
    # COMBINED OUTPUTS
    # --------------------------------------------------

    summary_df = pd.DataFrame(all_results)

    summary_df.to_csv(
        OUTPUT_DIR / "classwise_consistency_summary.csv",
        index=False,
    )

    combined_comparison = pd.concat(
        all_comparisons,
        ignore_index=True,
    )

    combined_comparison.to_csv(
        OUTPUT_DIR / "all_class_feature_comparisons.csv",
        index=False,
    )

    combined_pairwise = pd.concat(
        all_pairwise,
        ignore_index=True,
    )

    combined_pairwise.to_csv(
        OUTPUT_DIR / "all_class_jaccard_results.csv",
        index=False,
    )

    # --------------------------------------------------
    # FINAL SUMMARY
    # --------------------------------------------------

    print("\n" + "=" * 65)
    print("FINAL CLASS-WISE CONSISTENCY SUMMARY")
    print("=" * 65)

    print(
        summary_df.to_string(index=False)
    )

    print("\nClass-wise cross-fold analysis completed.")

    print("Results saved in:")
    print(OUTPUT_DIR)


if __name__ == "__main__":
    main()
