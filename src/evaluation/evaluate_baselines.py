from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.fi2010 import FI2010Loader
from src.data.sequences import create_sequences
from src.data.torch_dataset import FI2010TorchDataset

from src.models.baselines import (
    CNNOnly,
    GRUOnly,
    CNNGRUWithoutAttention
)

from src.models.multihorizon_model import TemporalCNNGRUAttention


# CONFIGURATION

DATASET_ROOT = (
    Path.home()
    / "Downloads"
    / "BenchmarkDatasets"
    / "NoAuction"
    / "1.NoAuction_Zscore"
)

CHECKPOINT_DIR = Path("results/checkpoints")
BASELINE_DIR = CHECKPOINT_DIR / "baselines"

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


# LOAD TEST DATA

print("Loading held-out test data...")

loader = FI2010Loader(str(DATASET_ROOT))

X_test, y_test = loader.load_fold(1, split="Testing")

X_seq, y_seq = create_sequences(
    X_test, y_test, sequence_length=100
)

test_dataset = FI2010TorchDataset(
    X_seq,
    y_seq,
    np.arange(len(X_seq))
)

test_loader = DataLoader(
    test_dataset,
    batch_size=8,
    shuffle=False,
    num_workers=0
)

print("Test sequences:", len(test_dataset))


# MODELS AND CHECKPOINTS

experiments = {
    "CNN Only": (
        CNNOnly,
        BASELINE_DIR / "best_cnn_only.pt"
    ),
    "GRU Only": (
        GRUOnly,
        BASELINE_DIR / "best_gru_only.pt"
    ),
    "CNN-GRU without Attention": (
        CNNGRUWithoutAttention,
        BASELINE_DIR / "best_cnn_gru_no_attention.pt"
    ),
    "CNN-GRU-Attention (Proposed)": (
        TemporalCNNGRUAttention,
        CHECKPOINT_DIR / "best_cf1_model.pt"
    )
}


# METRICS

def calculate_metrics(y_true, y_pred):

    scores = []

    for c in range(3):

        tp = np.sum((y_true == c) & (y_pred == c))
        fp = np.sum((y_true != c) & (y_pred == c))
        fn = np.sum((y_true == c) & (y_pred != c))

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        scores.append(f1)

    accuracy = np.mean(y_true == y_pred)

    return accuracy, float(np.mean(scores))


# EVALUATE ALL MODELS

results = {}

for name, (model_class, checkpoint_path) in experiments.items():

    print("\n" + "=" * 60)
    print("EVALUATING:", name)
    print("=" * 60)

    model = model_class().to(device)

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for batch_X, batch_y in test_loader:

            batch_X = batch_X.to(device)

            logits, _ = model(batch_X)

            all_predictions.append(
                logits.argmax(dim=-1).cpu().numpy()
            )

            all_targets.append(batch_y.numpy())

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    horizon_f1 = []
    horizon_accuracy = []

    for h in range(5):

        accuracy, macro_f1 = calculate_metrics(
            targets[:, h],
            predictions[:, h]
        )

        horizon_accuracy.append(accuracy)
        horizon_f1.append(macro_f1)

        print(
            f"Horizon {h + 1} | "
            f"Accuracy: {accuracy:.4f} | "
            f"Macro F1: {macro_f1:.4f}"
        )

    results[name] = {
        "accuracy": np.mean(horizon_accuracy),
        "macro_f1": np.mean(horizon_f1)
    }

# FINAL COMPARISON

print("\n" + "=" * 75)
print("FINAL HELD-OUT TEST COMPARISON")
print("=" * 75)

for name, metrics in results.items():

    print(
        f"{name:35s} | "
        f"Accuracy: {metrics['accuracy']:.4f} | "
        f"Macro F1: {metrics['macro_f1']:.4f}"
    )

print("\nBASELINE COMPARISON COMPLETED")
