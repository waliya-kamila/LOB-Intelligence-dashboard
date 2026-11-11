
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from src.data.fi2010 import FI2010Loader
from src.data.sequences import create_sequences
from src.models.multihorizon_model import TemporalCNNGRUAttention


# ==================================================
# CONFIGURATION
# ==================================================

FOLD = 3                # Change to 1, 2, or 3
SEQUENCE_LENGTH = 100
HORIZON_INDEX = 4        # Horizon 10
IG_STEPS = 128
SAMPLES_PER_CLASS = 24
BATCH_SIZE = 8
RANDOM_SEED = 42

NUM_FEATURES = 144
NUM_CLASSES = 3

PROJECT_ROOT = Path.cwd()

DATA_ROOT = Path(
    "~/Downloads/BenchmarkDatasets/NoAuction/1.NoAuction_Zscore"
).expanduser()

CHECKPOINT_PATH = (
    PROJECT_ROOT
    / f"results/checkpoints/multifold/fold{FOLD}_CNN-GRU-Attention.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / f"results/explainability/fold{FOLD}_integrated_gradients"
)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


# ==================================================
# INTEGRATED GRADIENTS FUNCTION
# ==================================================

def integrated_gradients(
    model,
    inputs,
    baseline,
    target_classes,
    horizon_index,
    steps=64,
):

    total_gradients = torch.zeros_like(inputs)

    alphas = torch.linspace(
        0.0, 1.0, steps + 1, device=inputs.device
    )

    for i, alpha in enumerate(alphas):

        interpolated = (
            baseline + alpha * (inputs - baseline)
        ).detach().requires_grad_(True)

        logits, _ = model(interpolated)

        selected_logits = logits[
            :, horizon_index, :
        ].gather(
            1, target_classes.unsqueeze(1)
        ).squeeze(1)

        gradients = torch.autograd.grad(
            outputs=selected_logits.sum(),
            inputs=interpolated,
        )[0]

        weight = (
            0.5 if i == 0 or i == steps else 1.0
        )

        total_gradients += weight * gradients

    average_gradients = total_gradients / steps

    attributions = (
        (inputs - baseline) * average_gradients
    )

    # Completeness error

    with torch.no_grad():

        input_logits, _ = model(inputs)
        baseline_logits, _ = model(baseline)

        input_scores = input_logits[
            :, horizon_index, :
        ].gather(
            1, target_classes.unsqueeze(1)
        ).squeeze(1)

        baseline_scores = baseline_logits[
            :, horizon_index, :
        ].gather(
            1, target_classes.unsqueeze(1)
        ).squeeze(1)

        score_difference = input_scores - baseline_scores

        attribution_sum = attributions.sum(dim=(1, 2))

        completeness_error = torch.abs(
            score_difference - attribution_sum
        )

    return attributions, completeness_error


# ==================================================
# MAIN FUNCTION
# ==================================================

def main():

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 55)
    print(f"INTEGRATED GRADIENTS - FOLD {FOLD}")
    print("=" * 55)

    print("Using device:", DEVICE)
    print("IG steps:", IG_STEPS)

    if not CHECKPOINT_PATH.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}"
        )

    # --------------------------------------------------
    # LOAD DATA
    # --------------------------------------------------

    loader = FI2010Loader(str(DATA_ROOT))

    X_train, y_train = loader.load_fold(
        FOLD, "Training"
    )

    X_test, y_test = loader.load_fold(
        FOLD, "Testing"
    )

    y_train = y_train.astype(np.int64) - 1
    y_test = y_test.astype(np.int64) - 1

    X_test_seq, y_test_seq = create_sequences(
        X_test,
        y_test,
        SEQUENCE_LENGTH,
    )

    print("Test sequence shape:", X_test_seq.shape)
    print("Test label shape:", y_test_seq.shape)

    # --------------------------------------------------
    # TRAINING BASELINE
    # --------------------------------------------------

    feature_mean = X_train.mean(
        axis=0
    ).astype(np.float32)

    baseline = torch.tensor(
        feature_mean,
        dtype=torch.float32,
        device=DEVICE,
    ).view(1, 1, NUM_FEATURES)

    # --------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------

    model = TemporalCNNGRUAttention(
        input_features=144,
        cnn_channels=64,
        gru_hidden=64,
        num_horizons=5,
        num_classes=3,
        dropout=0.2,
    ).to(DEVICE)

    checkpoint = torch.load(
        CHECKPOINT_PATH,
        map_location=DEVICE,
        weights_only=False,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print("Loaded checkpoint:", CHECKPOINT_PATH)
    print("Checkpoint epoch:", checkpoint.get("epoch"))

    # --------------------------------------------------
    # TEST PREDICTIONS
    # --------------------------------------------------

    X_tensor = torch.tensor(
        X_test_seq,
        dtype=torch.float32,
    )

    y_tensor = torch.tensor(
        y_test_seq,
        dtype=torch.long,
    )

    all_predictions = []

    with torch.no_grad():

        for start in range(
            0, len(X_tensor), BATCH_SIZE
        ):

            batch = X_tensor[
                start:start + BATCH_SIZE
            ].to(DEVICE)

            logits, _ = model(batch)

            predictions = logits.argmax(dim=-1)

            all_predictions.append(
                predictions.cpu().numpy()
            )

    predictions = np.concatenate(
        all_predictions,
        axis=0,
    )

    true_labels = y_tensor.numpy()

    # --------------------------------------------------
    # SELECT CORRECTLY CLASSIFIED SAMPLES
    # --------------------------------------------------

    rng = np.random.default_rng(RANDOM_SEED)

    selected_indices = []
    selected_classes = []

    for class_id in range(NUM_CLASSES):

        candidates = np.where(
            (predictions[:, HORIZON_INDEX] == class_id)
            &
            (true_labels[:, HORIZON_INDEX] == class_id)
        )[0]

        if len(candidates) == 0:
            raise RuntimeError(
                f"No correctly classified samples for class {class_id}"
            )

        count = min(
            SAMPLES_PER_CLASS,
            len(candidates),
        )

        chosen = rng.choice(
            candidates,
            size=count,
            replace=False,
        )

        selected_indices.extend(
            chosen.tolist()
        )

        selected_classes.extend(
            [class_id] * count
        )

        print(
            f"Class {class_id}: {count} samples "
            f"selected from {len(candidates)} correct predictions"
        )

    selected_indices = np.array(selected_indices)
    selected_classes = np.array(selected_classes)

    X_selected = X_tensor[selected_indices]

    y_selected = y_tensor[
        selected_indices,
        HORIZON_INDEX,
    ]

    # --------------------------------------------------
    # CALCULATE ATTRIBUTIONS
    # --------------------------------------------------

    all_attributions = []
    all_errors = []

    for start in range(
        0, len(X_selected), BATCH_SIZE
    ):

        end = start + BATCH_SIZE

        inputs = X_selected[
            start:end
        ].to(DEVICE)

        targets = y_selected[
            start:end
        ].to(DEVICE)

        batch_baseline = baseline.expand_as(inputs)

        attributions, errors = integrated_gradients(
            model=model,
            inputs=inputs,
            baseline=batch_baseline,
            target_classes=targets,
            horizon_index=HORIZON_INDEX,
            steps=IG_STEPS,
        )

        all_attributions.append(
            attributions.detach().cpu().numpy()
        )

        all_errors.append(
            errors.detach().cpu().numpy()
        )

        print(
            f"Processed {min(end, len(X_selected))}"
            f"/{len(X_selected)} samples"
        )

    attributions = np.concatenate(
        all_attributions,
        axis=0,
    )

    completeness_errors = np.concatenate(
        all_errors,
        axis=0,
    )

    print("\nAttribution shape:", attributions.shape)

    print(
        "Mean absolute completeness error:",
        completeness_errors.mean(),
    )

    # --------------------------------------------------
    # OVERALL FEATURE IMPORTANCE
    # --------------------------------------------------

    overall_importance = np.mean(
        np.abs(attributions),
        axis=(0, 1),
    )

    overall_signed = np.mean(
        attributions,
        axis=(0, 1),
    )

    results = pd.DataFrame({
        "feature_index": np.arange(NUM_FEATURES),
        "mean_absolute_attribution": overall_importance,
        "mean_signed_attribution": overall_signed,
    })

    results = results.sort_values(
        "mean_absolute_attribution",
        ascending=False,
    )

    results.to_csv(
        OUTPUT_DIR / "feature_importance_overall.csv",
        index=False,
    )

    # --------------------------------------------------
    # CLASS-WISE FEATURE IMPORTANCE
    # --------------------------------------------------

    class_results = []

    for class_id in range(NUM_CLASSES):

        class_mask = selected_classes == class_id

        class_attr = attributions[class_mask]

        class_importance = np.mean(
            np.abs(class_attr),
            axis=(0, 1),
        )

        for feature_id, importance in enumerate(
            class_importance
        ):

            class_results.append({
                "class_id": class_id,
                "feature_index": feature_id,
                "mean_absolute_attribution": importance,
            })

    class_df = pd.DataFrame(class_results)

    class_df.to_csv(
        OUTPUT_DIR / "feature_importance_by_class.csv",
        index=False,
    )

    # --------------------------------------------------
    # SAVE RAW RESULTS
    # --------------------------------------------------

    np.save(
        OUTPUT_DIR / "integrated_gradients_attributions.npy",
        attributions,
    )

    np.save(
        OUTPUT_DIR / "selected_sample_indices.npy",
        selected_indices,
    )

    np.save(
        OUTPUT_DIR / "completeness_errors.npy",
        completeness_errors,
    )

    # --------------------------------------------------
    # TOP 20 FEATURE VISUALIZATION
    # --------------------------------------------------

    top_features = results.head(20).sort_values(
        "mean_absolute_attribution",
        ascending=True,
    )

    plt.figure(figsize=(10, 8))

    plt.barh(
        top_features["feature_index"].astype(str),
        top_features["mean_absolute_attribution"],
    )

    plt.xlabel("Mean Absolute Integrated Gradients")
    plt.ylabel("Feature Index")

    plt.title(
        f"Top 20 FI-2010 Features - Fold {FOLD}\n"
        "Integrated Gradients | Horizon 10"
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "top20_feature_importance.png",
        dpi=300,
    )

    plt.close()

    # --------------------------------------------------
    # FINAL OUTPUT
    # --------------------------------------------------

    print("\nTOP 10 FEATURES")

    print(
        results.head(10).to_string(index=False)
    )

    print("\nIntegrated Gradients completed.")

    print("Fold:", FOLD)
    print("Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
