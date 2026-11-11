import numpy as np
from pathlib import Path

from src.data.fi2010 import FI2010Loader
from src.data.sequences import create_sequences
from src.data.splits import chronological_split


DATASET_ROOT = (
    Path.home()
    / "Downloads"
    / "BenchmarkDatasets"
    / "NoAuction"
    / "1.NoAuction_Zscore"
)


def macro_f1_score(y_true, y_pred):

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

    return np.mean(scores)


loader = FI2010Loader(str(DATASET_ROOT))

# Training data
X_train_raw, y_train_raw = loader.load_fold(
    1, split="Training"
)

X_train_seq, y_train_seq = create_sequences(
    X_train_raw, y_train_raw, sequence_length=100
)

_, y_train, _, _, _, _ = chronological_split(
    X_train_seq, y_train_seq
)

y_train = y_train.astype(np.int64) - 1

# Held-out test data
X_test_raw, y_test_raw = loader.load_fold(
    1, split="Testing"
)

X_test_seq, y_test = create_sequences(
    X_test_raw, y_test_raw, sequence_length=100
)

y_test = y_test.astype(np.int64) - 1

# Majority-class predictions
predictions = np.zeros_like(y_test)

print("\nMAJORITY CLASS BASELINE")
print("-" * 55)

scores = []

for h in range(5):

    counts = np.bincount(y_train[:, h], minlength=3)
    majority_class = int(np.argmax(counts))

    predictions[:, h] = majority_class

    accuracy = np.mean(predictions[:, h] == y_test[:, h])

    f1 = macro_f1_score(
        y_test[:, h],
        predictions[:, h]
    )

    scores.append(f1)

    print(
        f"Horizon {h + 1} | "
        f"Majority class: {majority_class} | "
        f"Accuracy: {accuracy:.4f} | "
        f"Macro F1: {f1:.4f}"
    )

print("\nOverall Baseline Macro F1:", round(np.mean(scores), 4))
print("BASELINE EVALUATION COMPLETED")
