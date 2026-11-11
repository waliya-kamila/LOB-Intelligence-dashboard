
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


# ============================================================
# CONFIGURATION
# ============================================================

FOLDS = [1, 2, 3]
HORIZONS = [1, 2, 3, 5, 10]
CLASS_NAMES = ["Up", "Stationary", "Down"]

BATCH_SIZE = 8
SEQUENCE_LENGTH = 100

DATASET_ROOT = (
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


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(y_true, y_pred):

    confusion = np.zeros((3, 3), dtype=np.int64)

    for actual, predicted in zip(y_true, y_pred):
        confusion[actual, predicted] += 1

    class_metrics = []

    for class_id in range(3):

        tp = confusion[class_id, class_id]
        fp = confusion[:, class_id].sum() - tp
        fn = confusion[class_id, :].sum() - tp

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        class_metrics.append({
            "class": CLASS_NAMES[class_id],
            "precision": precision,
            "recall": recall,
            "f1": f1
        })

    accuracy = np.trace(confusion) / confusion.sum()

    macro_precision = np.mean(
        [m["precision"] for m in class_metrics]
    )

    macro_recall = np.mean(
        [m["recall"] for m in class_metrics]
    )

    macro_f1 = np.mean(
        [m["f1"] for m in class_metrics]
    )

    return {
        "accuracy": accuracy,
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "class_metrics": class_metrics,
        "confusion_matrix": confusion
    }


# ============================================================
# PREDICTION
# ============================================================

def get_predictions(model, test_loader):

    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for X, y in test_loader:

            X = X.to(device)

            logits, _ = model(X)

            predictions = logits.argmax(dim=-1)

            all_predictions.append(
                predictions.cpu().numpy()
            )

            all_targets.append(
                y.numpy()
            )

    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)

    return predictions, targets


# ============================================================
# LOAD TEST DATA
# ============================================================

print("=" * 65)
print("MULTI-FOLD DETAILED EVALUATION")
print("=" * 65)

print("Evaluation device:", device)

loader = FI2010Loader(str(DATASET_ROOT))

all_results = []
all_confusion_matrices = []


# ============================================================
# MAIN EVALUATION
# ============================================================

for fold in FOLDS:

    print("\n" + "=" * 65)
    print("FOLD:", fold)
    print("=" * 65)

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

    print("Test sequences:", len(test_dataset))

    for model_name, model_class in MODELS.items():

        print("\nModel:", model_name)

        checkpoint_path = (
            CHECKPOINT_DIR
            / f"fold{fold}_{model_name}.pt"
        )

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint missing: {checkpoint_path}"
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

        predictions, targets = get_predictions(
            model,
            test_loader
        )

        print(
            "Checkpoint epoch:",
            checkpoint["epoch"]
        )

        # ----------------------------------------------------
        # PER-HORIZON AND PER-CLASS METRICS
        # ----------------------------------------------------

        for h, horizon in enumerate(HORIZONS):

            metrics = calculate_metrics(
                targets[:, h],
                predictions[:, h]
            )

            print(
                f"\nHorizon {horizon}: "
                f"Accuracy={metrics['accuracy']:.4f}, "
                f"Macro F1={metrics['macro_f1']:.4f}"
            )

            all_results.append({
                "fold": fold,
                "model": model_name,
                "horizon": horizon,
                "accuracy": metrics["accuracy"],
                "macro_precision": metrics["macro_precision"],
                "macro_recall": metrics["macro_recall"],
                "macro_f1": metrics["macro_f1"]
            })

            for class_metric in metrics["class_metrics"]:

                all_results.append({
                    "fold": fold,
                    "model": model_name,
                    "horizon": horizon,
                    "accuracy": "",
                    "macro_precision": "",
                    "macro_recall": "",
                    "macro_f1": "",
                    "class": class_metric["class"],
                    "class_precision": class_metric["precision"],
                    "class_recall": class_metric["recall"],
                    "class_f1": class_metric["f1"]
                })

            # Save confusion matrix
            confusion = metrics["confusion_matrix"]

            for actual_id in range(3):
                for predicted_id in range(3):

                    all_confusion_matrices.append({
                        "fold": fold,
                        "model": model_name,
                        "horizon": horizon,
                        "actual_class": CLASS_NAMES[actual_id],
                        "predicted_class": CLASS_NAMES[predicted_id],
                        "count": int(
                            confusion[actual_id, predicted_id]
                        )
                    })


# ============================================================
# SAVE DETAILED METRICS
# ============================================================

metrics_path = OUTPUT_DIR / "detailed_metrics.csv"

fieldnames = [
    "fold",
    "model",
    "horizon",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "class",
    "class_precision",
    "class_recall",
    "class_f1"
]

with open(metrics_path, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=fieldnames
    )

    writer.writeheader()
    writer.writerows(all_results)


# ============================================================
# SAVE CONFUSION MATRICES
# ============================================================

confusion_path = OUTPUT_DIR / "confusion_matrices.csv"

with open(confusion_path, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "fold",
            "model",
            "horizon",
            "actual_class",
            "predicted_class",
            "count"
        ]
    )

    writer.writeheader()
    writer.writerows(all_confusion_matrices)


# ============================================================
# SUMMARY: MEAN AND STANDARD DEVIATION
# ============================================================

summary_path = OUTPUT_DIR / "horizon_summary.csv"

summary_rows = []

for model_name in MODELS:

    for horizon in HORIZONS:

        scores = [
            row["macro_f1"]
            for row in all_results
            if row["model"] == model_name
            and row["horizon"] == horizon
            and row["macro_f1"] != ""
        ]

        summary_rows.append({
            "model": model_name,
            "horizon": horizon,
            "mean_macro_f1": np.mean(scores),
            "std_macro_f1": np.std(scores, ddof=1)
        })

with open(summary_path, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "model",
            "horizon",
            "mean_macro_f1",
            "std_macro_f1"
        ]
    )

    writer.writeheader()
    writer.writerows(summary_rows)


# ============================================================
# FINAL OUTPUT
# ============================================================

print("\n" + "=" * 65)
print("MULTI-FOLD EVALUATION COMPLETED")
print("=" * 65)

print("\nHORIZON-WISE MEAN MACRO F1")

for row in summary_rows:

    print(
        f"{row['model']} | "
        f"Horizon {row['horizon']} | "
        f"Mean F1: {row['mean_macro_f1']:.4f} | "
        f"Std: {row['std_macro_f1']:.4f}"
    )

print("\nFiles saved:")
print(metrics_path)
print(confusion_path)
print(summary_path)
