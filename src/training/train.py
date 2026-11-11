from pathlib import Path
import random
import numpy as np
import torch
import torch.nn as nn

from src.data.fi2010 import FI2010Loader
from src.data.sequences import create_sequences
from src.data.splits import chronological_split
from src.data.class_weights import calculate_class_weights
from src.data.torch_dataset import create_dataloaders
from src.models.multihorizon_model import TemporalCNNGRUAttention
from src.training.losses import MultiHorizonLoss


# ---------------- CONFIGURATION ----------------

SEED = 42
FOLD = 1
SEQUENCE_LENGTH = 100
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
PATIENCE = 5

DATASET_ROOT = (
    Path.home()
    / "Downloads"
    / "BenchmarkDatasets"
    / "NoAuction"
    / "1.NoAuction_Zscore"
)

CHECKPOINT_DIR = Path("results/checkpoints")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------- REPRODUCIBILITY ----------------

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("\nDEVICE:", device)


# ---------------- METRICS ----------------

def calculate_macro_f1(predictions, targets):

    horizon_scores = []

    for h in range(5):

        y_true = targets[:, h]
        y_pred = predictions[:, h]

        class_scores = []

        for class_id in range(3):

            tp = np.sum((y_true == class_id) & (y_pred == class_id))
            fp = np.sum((y_true != class_id) & (y_pred == class_id))
            fn = np.sum((y_true == class_id) & (y_pred != class_id))

            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0

            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )

            class_scores.append(f1)

        horizon_scores.append(np.mean(class_scores))

    return float(np.mean(horizon_scores))


# ---------------- LOAD DATA ----------------

print("\nLoading FI-2010 dataset...")

loader = FI2010Loader(str(DATASET_ROOT))

X, y = loader.load_fold(FOLD, split="Training")

print("Raw X:", X.shape)
print("Raw y:", y.shape)


# ---------------- CREATE SEQUENCES ----------------

print("\nCreating sequences...")

X_seq, y_seq = create_sequences(
    X,
    y,
    sequence_length=SEQUENCE_LENGTH
)

print("Sequence X:", X_seq.shape)
print("Sequence y:", y_seq.shape)


# ---------------- CHRONOLOGICAL SPLIT ----------------

print("\nCreating chronological split...")

(
    X_train,
    y_train,
    X_val,
    y_val,
    train_indices,
    val_indices
) = chronological_split(X_seq, y_seq)

print("Training samples:", len(train_indices))
print("Validation samples:", len(val_indices))


# ---------------- CLASS WEIGHTS ----------------

# Original labels: 1, 2, 3
# Mapped labels:   0, 1, 2

y_train_mapped = y_train.astype(np.int64) - 1

class_weights = calculate_class_weights(y_train_mapped)

print("\nClass weights shape:", class_weights.shape)


# ---------------- DATALOADERS ----------------

train_loader, val_loader = create_dataloaders(
    X_seq,
    y_seq,
    train_indices,
    val_indices,
    batch_size=BATCH_SIZE
)


# ---------------- MODEL AND LOSS ----------------

model = TemporalCNNGRUAttention().to(device)

criterion = MultiHorizonLoss(class_weights).to(device)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)

print("\nModel parameters:", sum(
    p.numel() for p in model.parameters()
))


# ---------------- TRAINING LOOP ----------------

best_val_loss = float("inf")
epochs_without_improvement = 0

checkpoint_path = CHECKPOINT_DIR / "best_cf1_model.pt"

for epoch in range(EPOCHS):

    model.train()

    train_loss_sum = 0.0
    train_samples = 0

    for batch_X, batch_y in train_loader:

        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad(set_to_none=True)

        predictions, attention = model(batch_X)

        loss = criterion(predictions, batch_y)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0
        )

        optimizer.step()

        batch_size = batch_X.size(0)

        train_loss_sum += loss.item() * batch_size
        train_samples += batch_size

    train_loss = train_loss_sum / train_samples

    # ---------------- VALIDATION ----------------

    model.eval()

    val_loss_sum = 0.0
    val_samples = 0

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for batch_X, batch_y in val_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            predictions, attention = model(batch_X)

            loss = criterion(predictions, batch_y)

            batch_size = batch_X.size(0)

            val_loss_sum += loss.item() * batch_size
            val_samples += batch_size

            predicted_classes = predictions.argmax(dim=-1)

            all_predictions.append(
                predicted_classes.cpu().numpy()
            )

            all_targets.append(
                batch_y.cpu().numpy()
            )

    val_loss = val_loss_sum / val_samples

    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    val_macro_f1 = calculate_macro_f1(
        all_predictions,
        all_targets
    )

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Macro F1: {val_macro_f1:.4f}"
    )

    # ---------------- SAVE BEST MODEL ----------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss
        epochs_without_improvement = 0

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": {
                    k: v.detach().cpu()
                    for k, v in model.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_macro_f1": val_macro_f1,
                "config": {
                    "fold": FOLD,
                    "sequence_length": SEQUENCE_LENGTH,
                    "batch_size": BATCH_SIZE,
                    "learning_rate": LEARNING_RATE
                }
            },
            checkpoint_path
        )

        print("Best checkpoint saved.")

    else:

        epochs_without_improvement += 1

    # ---------------- EARLY STOPPING ----------------

    if epochs_without_improvement >= PATIENCE:

        print("\nEarly stopping triggered.")
        break


print("\nTRAINING COMPLETED")
print("Best validation loss:", best_val_loss)
print("Checkpoint:", checkpoint_path)
