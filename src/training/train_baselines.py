from pathlib import Path
import random
import numpy as np
import torch

from src.data.fi2010 import FI2010Loader
from src.data.sequences import create_sequences
from src.data.splits import chronological_split
from src.data.class_weights import calculate_class_weights
from src.data.torch_dataset import create_dataloaders

from src.models.baselines import (
    CNNOnly,
    GRUOnly,
    CNNGRUWithoutAttention
)

from src.training.losses import MultiHorizonLoss


# CONFIGURATION

SEED = 42
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

CHECKPOINT_DIR = Path("results/checkpoints/baselines")
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


# REPRODUCIBILITY

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# MACRO F1

def calculate_macro_f1(y_true, y_pred):

    scores = []

    for h in range(5):

        class_scores = []

        for c in range(3):

            tp = np.sum(
                (y_true[:, h] == c) & (y_pred[:, h] == c)
            )
            fp = np.sum(
                (y_true[:, h] != c) & (y_pred[:, h] == c)
            )
            fn = np.sum(
                (y_true[:, h] == c) & (y_pred[:, h] != c)
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


# LOAD DATA ONCE

print("DEVICE:", device)
print("\nLoading FI-2010...")

loader = FI2010Loader(str(DATASET_ROOT))

X, y = loader.load_fold(1, split="Training")

X_seq, y_seq = create_sequences(
    X, y, sequence_length=100
)

(
    X_train,
    y_train,
    X_val,
    y_val,
    train_indices,
    val_indices
) = chronological_split(X_seq, y_seq)

y_train_mapped = y_train.astype(np.int64) - 1

class_weights = calculate_class_weights(y_train_mapped)

train_loader, val_loader = create_dataloaders(
    X_seq,
    y_seq,
    train_indices,
    val_indices,
    batch_size=BATCH_SIZE
)

print("Training samples:", len(train_indices))
print("Validation samples:", len(val_indices))


# MODEL DEFINITIONS

model_classes = {
    "cnn_only": CNNOnly,
    "gru_only": GRUOnly,
    "cnn_gru_no_attention": CNNGRUWithoutAttention
}


# TRAINING FUNCTION

def train_model(model_name, model_class):

    print("\n" + "=" * 60)
    print("TRAINING:", model_name)
    print("=" * 60)

    set_seed(SEED)

    model = model_class().to(device)

    criterion = MultiHorizonLoss(class_weights).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_val_loss = float("inf")
    patience_counter = 0

    checkpoint_path = (
        CHECKPOINT_DIR / f"best_{model_name}.pt"
    )

    for epoch in range(EPOCHS):

        # TRAINING

        model.train()

        train_loss_sum = 0
        train_samples = 0

        for batch_X, batch_y in train_loader:

            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)

            predictions, _ = model(batch_X)

            loss = criterion(predictions, batch_y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

            size = batch_X.size(0)

            train_loss_sum += loss.item() * size
            train_samples += size

        train_loss = train_loss_sum / train_samples

        # VALIDATION

        model.eval()

        val_loss_sum = 0
        val_samples = 0

        all_predictions = []
        all_targets = []

        with torch.no_grad():

            for batch_X, batch_y in val_loader:

                batch_X = batch_X.to(device)
                batch_y = batch_y.to(device)

                predictions, _ = model(batch_X)

                loss = criterion(predictions, batch_y)

                size = batch_X.size(0)

                val_loss_sum += loss.item() * size
                val_samples += size

                all_predictions.append(
                    predictions.argmax(dim=-1).cpu().numpy()
                )

                all_targets.append(
                    batch_y.cpu().numpy()
                )

        val_loss = val_loss_sum / val_samples

        predictions_np = np.concatenate(
            all_predictions, axis=0
        )

        targets_np = np.concatenate(
            all_targets, axis=0
        )

        val_f1 = calculate_macro_f1(
            targets_np, predictions_np
        )

        print(
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Macro F1: {val_f1:.4f}"
        )

        # SAVE BEST CHECKPOINT

        if val_loss < best_val_loss:

            best_val_loss = val_loss
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": {
                        k: v.detach().cpu()
                        for k, v in model.state_dict().items()
                    },
                    "epoch": epoch + 1,
                    "val_loss": val_loss,
                    "val_macro_f1": val_f1,
                    "model_name": model_name
                },
                checkpoint_path
            )

            print("Best checkpoint saved.")

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print("Early stopping triggered.")
            break

    print("\nCompleted:", model_name)
    print("Best validation loss:", best_val_loss)
    print("Checkpoint:", checkpoint_path)


# RUN ALL BASELINES

for name, model_class in model_classes.items():
    train_model(name, model_class)


print("\n" + "=" * 60)
print("ALL BASELINE TRAINING COMPLETED")
print("=" * 60)
