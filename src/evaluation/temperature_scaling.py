
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader

from src.data.fi2010 import FI2010Loader
from src.data.sequences import create_sequences
from src.data.splits import chronological_split
from src.data.torch_dataset import (
    create_dataloaders,
    FI2010TorchDataset
)

from src.models.baselines import CNNGRUWithoutAttention
from src.models.multihorizon_model import TemporalCNNGRUAttention


# ============================================================
# CONFIGURATION
# ============================================================

FOLDS = [1, 2, 3]
HORIZON_INDEX = 4
HORIZON = 10

BATCH_SIZE = 8
SEQUENCE_LENGTH = 100
N_BINS = 15

ROOT = (
    Path.home()
    / "Downloads"
    / "BenchmarkDatasets"
    / "NoAuction"
    / "1.NoAuction_Zscore"
)

CHECKPOINT_DIR = Path("results/checkpoints/multifold")

OUTPUT_DIR = Path("results/evaluation_multifold")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

MODELS = {
    "CNN-GRU-NoAttention": CNNGRUWithoutAttention,
    "CNN-GRU-Attention": TemporalCNNGRUAttention
}

PROB_COLS = [
    "prob_up",
    "prob_stationary",
    "prob_down"
]

CLASS_NAMES = ["Up", "Stationary", "Down"]


# ============================================================
# EXTRACT LOGITS
# ============================================================

def extract_logits(model, data_loader):

    model.eval()

    all_logits = []
    all_targets = []

    with torch.no_grad():

        for X, y in data_loader:

            X = X.to(device)

            logits, _ = model(X)

            all_logits.append(
                logits[:, HORIZON_INDEX, :].cpu()
            )

            all_targets.append(
                y[:, HORIZON_INDEX].cpu()
            )

    return (
        torch.cat(all_logits),
        torch.cat(all_targets)
    )


# ============================================================
# FIT TEMPERATURE USING VALIDATION DATA
# ============================================================

def fit_temperature(val_logits, val_targets):

    log_temperature = torch.nn.Parameter(
        torch.zeros(1, dtype=torch.float64)
    )

    logits = val_logits.double()
    targets = val_targets.long()

    optimizer = torch.optim.LBFGS(
        [log_temperature],
        lr=0.1,
        max_iter=100,
        line_search_fn="strong_wolfe"
    )

    def closure():

        optimizer.zero_grad()

        temperature = torch.exp(log_temperature)

        loss = F.cross_entropy(
            logits / temperature,
            targets
        )

        loss.backward()

        return loss

    optimizer.step(closure)

    temperature = torch.exp(
        log_temperature.detach()
    ).item()

    return temperature


# ============================================================
# CALIBRATION METRICS
# ============================================================

def calculate_metrics(probabilities, targets):

    targets = np.asarray(targets, dtype=np.int64)

    predicted = probabilities.argmax(axis=1)

    confidence = probabilities.max(axis=1)

    correct = (predicted == targets).astype(float)

    accuracy = correct.mean()

    # Multiclass Brier Score
    one_hot = np.eye(3)[targets]

    brier = np.mean(
        np.sum((probabilities - one_hot) ** 2, axis=1)
    )

    # Negative Log Likelihood
    true_probabilities = probabilities[
        np.arange(len(targets)), targets
    ]

    nll = -np.mean(
        np.log(np.clip(true_probabilities, 1e-12, 1.0))
    )

    # Expected Calibration Error
    ece = 0.0

    bin_edges = np.linspace(0, 1, N_BINS + 1)

    for i in range(N_BINS):

        lower = bin_edges[i]
        upper = bin_edges[i + 1]

        if i == N_BINS - 1:
            mask = (
                (confidence >= lower)
                & (confidence <= upper)
            )
        else:
            mask = (
                (confidence >= lower)
                & (confidence < upper)
            )

        if mask.sum() == 0:
            continue

        bin_accuracy = correct[mask].mean()
        bin_confidence = confidence[mask].mean()

        ece += (
            mask.sum() / len(targets)
        ) * abs(bin_accuracy - bin_confidence)

    return {
        "accuracy": accuracy,
        "ECE": ece,
        "Brier_score": brier,
        "NLL": nll,
        "mean_confidence": confidence.mean()
    }


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    loader = FI2010Loader(str(ROOT))

    all_predictions = []
    all_metrics = []
    temperature_rows = []

    print("=" * 65)
    print("TEMPERATURE SCALING - HORIZON 10")
    print("=" * 65)

    print("Device:", device)

    for fold in FOLDS:

        print("\n" + "=" * 65)
        print("FOLD:", fold)
        print("=" * 65)

        # Load training portion to reconstruct validation split
        X_train_raw, y_train_raw = loader.load_fold(
            fold,
            split="Training"
        )

        X_seq, y_seq = create_sequences(
            X_train_raw,
            y_train_raw,
            sequence_length=SEQUENCE_LENGTH
        )

        (
            X_train,
            y_train,
            X_val,
            y_val,
            train_indices,
            val_indices
        ) = chronological_split(X_seq, y_seq)

        _, val_loader = create_dataloaders(
            X_seq,
            y_seq,
            train_indices,
            val_indices,
            batch_size=BATCH_SIZE
        )

        # Load test portion
        X_test_raw, y_test_raw = loader.load_fold(
            fold,
            split="Testing"
        )

        X_test_seq, y_test_seq = create_sequences(
            X_test_raw,
            y_test_raw,
            sequence_length=SEQUENCE_LENGTH
        )

        test_dataset = FI2010TorchDataset(
            X_test_seq,
            y_test_seq,
            np.arange(len(X_test_seq))
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0
        )

        for model_name, model_class in MODELS.items():

            print("\nModel:", model_name)

            checkpoint_path = (
                CHECKPOINT_DIR
                / f"fold{fold}_{model_name}.pt"
            )

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True
            )

            model = model_class().to(device)

            model.load_state_dict(
                checkpoint["model_state_dict"]
            )

            # Validation logits
            val_logits, val_targets = extract_logits(
                model,
                val_loader
            )

            # Learn temperature using validation only
            temperature = fit_temperature(
                val_logits,
                val_targets
            )

            print(
                f"Learned temperature: {temperature:.5f}"
            )

            # Test logits
            test_logits, test_targets = extract_logits(
                model,
                test_loader
            )

            # Original test probabilities
            original_probs = torch.softmax(
                test_logits,
                dim=1
            ).numpy()

            # Temperature-scaled probabilities
            calibrated_probs = torch.softmax(
                test_logits / temperature,
                dim=1
            ).numpy()

            targets = test_targets.numpy()

            original_metrics = calculate_metrics(
                original_probs,
                targets
            )

            calibrated_metrics = calculate_metrics(
                calibrated_probs,
                targets
            )

            print("\nOriginal metrics:")
            print({
                k: round(v, 5)
                for k, v in original_metrics.items()
            })

            print("\nCalibrated metrics:")
            print({
                k: round(v, 5)
                for k, v in calibrated_metrics.items()
            })

            temperature_rows.append({
                "fold": fold,
                "model": model_name,
                "temperature": temperature
            })

            for stage, metrics in [
                ("Original", original_metrics),
                ("Calibrated", calibrated_metrics)
            ]:

                all_metrics.append({
                    "fold": fold,
                    "model": model_name,
                    "stage": stage,
                    "temperature": temperature,
                    **metrics
                })

            # Save individual calibrated probabilities
            for i in range(len(targets)):

                for stage, probs in [
                    ("Original", original_probs),
                    ("Calibrated", calibrated_probs)
                ]:

                    all_predictions.append({
                        "fold": fold,
                        "model": model_name,
                        "stage": stage,
                        "actual_class": CLASS_NAMES[targets[i]],
                        "predicted_class": CLASS_NAMES[
                            probs[i].argmax()
                        ],
                        "prob_up": probs[i, 0],
                        "prob_stationary": probs[i, 1],
                        "prob_down": probs[i, 2],
                        "confidence": probs[i].max(),
                        "correct": int(
                            probs[i].argmax() == targets[i]
                        )
                    })

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    metrics_df = pd.DataFrame(all_metrics)

    predictions_df = pd.DataFrame(all_predictions)

    temperatures_df = pd.DataFrame(temperature_rows)

    metrics_path = (
        OUTPUT_DIR / "temperature_scaling_metrics.csv"
    )

    predictions_path = (
        OUTPUT_DIR / "temperature_scaling_predictions.csv"
    )

    temperatures_path = (
        OUTPUT_DIR / "learned_temperatures.csv"
    )

    metrics_df.to_csv(metrics_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    temperatures_df.to_csv(temperatures_path, index=False)

    # Pooled test predictions, using each fold's own temperature
    pooled_summary = metrics_df.groupby(
        ["model", "stage"]
    )[
        ["accuracy", "ECE", "Brier_score", "NLL",
         "mean_confidence"]
    ].mean().reset_index()

    summary_path = (
        OUTPUT_DIR / "temperature_scaling_summary.csv"
    )

    pooled_summary.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("TEMPERATURE SCALING SUMMARY")
    print("=" * 65)

    print(pooled_summary.round(5).to_string(index=False))

    print("\nLEARNED TEMPERATURES")
    print(temperatures_df.round(5).to_string(index=False))

    print("\nSaved files:")
    print(metrics_path)
    print(predictions_path)
    print(temperatures_path)
    print(summary_path)


if __name__ == "__main__":
    main()
