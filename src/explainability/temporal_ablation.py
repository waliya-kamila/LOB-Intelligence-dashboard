
from pathlib import Path
import csv
import numpy as np
import torch

from torch.utils.data import DataLoader

from src.data.fi2010 import FI2010Loader
from src.data.sequences import create_sequences
from src.data.torch_dataset import FI2010TorchDataset

from src.models.baselines import CNNGRUWithoutAttention
from src.models.multihorizon_model import TemporalCNNGRUAttention


# CONFIGURATION
FOLD = 1
BATCH_SIZE = 8
SEQUENCE_LENGTH = 100

ROOT = (
    Path.home()
    / "Downloads"
    / "BenchmarkDatasets"
    / "NoAuction"
    / "1.NoAuction_Zscore"
)

CHECKPOINT_DIR = Path("results/checkpoints/multifold")
OUTPUT_DIR = Path("results/explainability")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

MODELS = {
    "NoAttention": CNNGRUWithoutAttention,
    "Attention": TemporalCNNGRUAttention
}

REGIONS = {
    "Original": None,
    "Early_0_25": (0, 25),
    "Middle_25_75": (25, 75),
    "Recent_75_100": (75, 100)
}


# MACRO F1
def macro_f1(y_true, y_pred):
    scores = []

    for h in range(5):
        class_scores = []

        for c in range(3):
            tp = np.sum(
                (y_true[:, h] == c) &
                (y_pred[:, h] == c)
            )
            fp = np.sum(
                (y_true[:, h] != c) &
                (y_pred[:, h] == c)
            )
            fn = np.sum(
                (y_true[:, h] == c) &
                (y_pred[:, h] != c)
            )

            precision = tp / (tp + fp) if tp + fp else 0
            recall = tp / (tp + fn) if tp + fn else 0

            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0
            )

            class_scores.append(f1)

        scores.append(np.mean(class_scores))

    return float(np.mean(scores))


# LOAD DATA
print("Loading data...")

loader = FI2010Loader(str(ROOT))

X_train, _ = loader.load_fold(FOLD, split="Training")
X_test, y_test = loader.load_fold(FOLD, split="Testing")

# Feature-wise training mean
feature_mean = X_train.mean(axis=0)

X_seq, y_seq = create_sequences(
    X_test,
    y_test,
    sequence_length=SEQUENCE_LENGTH
)

dataset = FI2010TorchDataset(
    X_seq,
    y_seq,
    np.arange(len(X_seq))
)

test_loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Test sequences:", len(dataset))


# EXPERIMENT
results = []

for model_name, model_class in MODELS.items():

    print("\nMODEL:", model_name)

    checkpoint_path = (
        CHECKPOINT_DIR /
        f"fold{FOLD}_CNN-GRU-{model_name}.pt"
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model = model_class().to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    original_predictions = []
    all_targets = []

    # Original predictions
    with torch.no_grad():
        for X, y in test_loader:
            logits, _ = model(X.to(device))

            original_predictions.append(
                logits.argmax(dim=-1).cpu().numpy()
            )
            all_targets.append(y.numpy())

    original_predictions = np.concatenate(
        original_predictions
    )
    targets = np.concatenate(all_targets)

    for region_name, bounds in REGIONS.items():

        print("Testing:", region_name)

        predictions = []

        with torch.no_grad():

            for X, y in test_loader:

                X = X.clone()

                if bounds is not None:
                    start, end = bounds

                    X[:, start:end, :] = torch.as_tensor(
                        feature_mean,
                        dtype=X.dtype
                    )

                logits, _ = model(X.to(device))

                predictions.append(
                    logits.argmax(dim=-1).cpu().numpy()
                )

        predictions = np.concatenate(predictions)

        score = macro_f1(targets, predictions)

        flip_rate = np.mean(
            predictions != original_predictions
        )

        results.append({
            "model": model_name,
            "region": region_name,
            "macro_f1": score,
            "prediction_flip_rate": float(flip_rate)
        })

        print(
            f"Macro F1: {score:.4f} | "
            f"Prediction changes: {flip_rate:.4f}"
        )


# SAVE RESULTS
output_file = OUTPUT_DIR / "temporal_ablation.csv"

with open(output_file, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "region",
            "macro_f1",
            "prediction_flip_rate"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

print("\nTEMPORAL ABLATION COMPLETED")
print("Results saved:", output_file)
