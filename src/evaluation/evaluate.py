from pathlib import Path
import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.fi2010 import FI2010Loader
from src.data.sequences import create_sequences
from src.data.torch_dataset import FI2010TorchDataset
from src.models.multihorizon_model import TemporalCNNGRUAttention


# Configuration
FOLD = 1
BATCH_SIZE = 8

DATASET_ROOT = (
    Path.home()
    / "Downloads"
    / "BenchmarkDatasets"
    / "NoAuction"
    / "1.NoAuction_Zscore"
)

CHECKPOINT_PATH = Path(
    "results/checkpoints/best_cf1_model.pt"
)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

print("Evaluation device:", device)


# Load held-out test data
print("\nLoading test dataset...")

loader = FI2010Loader(str(DATASET_ROOT))
X_test, y_test = loader.load_fold(FOLD, split="Testing")

print("Raw test X:", X_test.shape)
print("Raw test y:", y_test.shape)


# Create sequences
X_seq, y_seq = create_sequences(
    X_test,
    y_test,
    sequence_length=100
)

print("Test sequences:", X_seq.shape)


# Test DataLoader
test_indices = np.arange(len(X_seq))

test_dataset = FI2010TorchDataset(
    X_seq,
    y_seq,
    test_indices
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# Load trained model
model = TemporalCNNGRUAttention().to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

print("\nLoaded checkpoint from epoch:", checkpoint["epoch"])
print("Checkpoint validation loss:", checkpoint["val_loss"])


# Evaluation
all_predictions = []
all_targets = []

with torch.no_grad():

    for batch_X, batch_y in test_loader:

        batch_X = batch_X.to(device)

        predictions, attention = model(batch_X)

        predicted_classes = predictions.argmax(dim=-1)

        all_predictions.append(
            predicted_classes.cpu().numpy()
        )

        all_targets.append(
            batch_y.numpy()
        )


predictions = np.concatenate(all_predictions, axis=0)
targets = np.concatenate(all_targets, axis=0)


# Metrics
def calculate_metrics(y_true, y_pred):

    confusion = np.zeros((3, 3), dtype=np.int64)

    for actual, predicted in zip(y_true, y_pred):
        confusion[actual, predicted] += 1

    class_f1 = []

    for class_id in range(3):

        tp = confusion[class_id, class_id]
        fp = confusion[:, class_id].sum() - tp
        fn = confusion[class_id, :].sum() - tp

        precision = tp / (tp + fp) if tp + fp else 0
        recall = tp / (tp + fn) if tp + fn else 0

        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0
        )

        class_f1.append(f1)

    accuracy = np.trace(confusion) / confusion.sum()
    macro_f1 = np.mean(class_f1)

    return accuracy, macro_f1, confusion, class_f1


print("\n" + "=" * 65)
print("HELD-OUT TEST RESULTS")
print("=" * 65)

horizon_scores = []

for h in range(5):

    accuracy, macro_f1, confusion, class_f1 = calculate_metrics(
        targets[:, h],
        predictions[:, h]
    )

    horizon_scores.append(macro_f1)

    print(f"\nHorizon {h + 1} (offset: {[1, 2, 3, 5, 10][h]})")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Macro F1: {macro_f1:.4f}")
    print("Class F1 (Up, Stationary, Down):",
          np.round(class_f1, 4))

    print("Confusion Matrix:")
    print(confusion)


print("\n" + "=" * 65)
print("OVERALL TEST MACRO F1:", round(np.mean(horizon_scores), 4))
print("TOTAL TEST SEQUENCES:", len(targets))
print("EVALUATION COMPLETED")
