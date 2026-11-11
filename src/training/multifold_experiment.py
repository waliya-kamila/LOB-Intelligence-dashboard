from pathlib import Path
import csv
import random
import numpy as np
import torch

from torch.utils.data import DataLoader

from src.data.fi2010 import FI2010Loader
from src.data.sequences import create_sequences
from src.data.splits import chronological_split
from src.data.class_weights import calculate_class_weights
from src.data.torch_dataset import create_dataloaders, FI2010TorchDataset

from src.models.baselines import CNNGRUWithoutAttention
from src.models.multihorizon_model import TemporalCNNGRUAttention
from src.training.losses import MultiHorizonLoss


# CONFIGURATION

FOLDS = [1, 2, 3]
EPOCHS = 20
BATCH_SIZE = 8
LEARNING_RATE = 0.001
WEIGHT_DECAY = 0.0001
PATIENCE = 5

ROOT = (
    Path.home()
    / "Downloads"
    / "BenchmarkDatasets"
    / "NoAuction"
    / "1.NoAuction_Zscore"
)

OUTPUT_DIR = Path("results/checkpoints/multifold")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

MODELS = {
    "CNN-GRU-NoAttention": CNNGRUWithoutAttention,
    "CNN-GRU-Attention": TemporalCNNGRUAttention
}


# METRIC

def macro_f1(y_true, y_pred):

    scores = []

    for h in range(5):
        class_scores = []

        for c in range(3):

            tp = np.sum((y_true[:, h] == c) & (y_pred[:, h] == c))
            fp = np.sum((y_true[:, h] != c) & (y_pred[:, h] == c))
            fn = np.sum((y_true[:, h] == c) & (y_pred[:, h] != c))

            precision = tp / (tp + fp) if tp + fp else 0
            recall = tp / (tp + fn) if tp + fn else 0

            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0
            )

            class_scores.append(f1)

        scores.append(np.mean(class_scores))

    return float(np.mean(scores))


def evaluate(model, loader):

    model.eval()

    total_loss = 0
    total_samples = 0
    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(device)
            y = y.to(device)

            logits, _ = model(X)

            predictions = logits.argmax(dim=-1)

            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(y.cpu().numpy())

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    return macro_f1(targets, predictions)


# TRAIN ONE MODEL ON ONE FOLD

def train_one_model(model_name, model_class, fold, train_loader,
                    val_loader, class_weights):

    seed = 42 + fold

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    model = model_class().to(device)

    criterion = MultiHorizonLoss(class_weights).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY
    )

    best_val_f1 = -1.0
    patience_count = 0

    checkpoint_path = (
        OUTPUT_DIR / f"fold{fold}_{model_name}.pt"
    )

    for epoch in range(EPOCHS):

        model.train()

        for X, y in train_loader:

            X = X.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits, _ = model(X)
            loss = criterion(logits, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), max_norm=1.0
            )

            optimizer.step()

        val_f1 = evaluate(model, val_loader)

        print(
            f"Fold {fold} | {model_name} | "
            f"Epoch {epoch + 1}/{EPOCHS} | "
            f"Validation Macro F1: {val_f1:.4f}"
        )

        if val_f1 > best_val_f1:

            best_val_f1 = val_f1
            patience_count = 0

            torch.save(
                {
                    "model_state_dict": {
                        k: v.detach().cpu()
                        for k, v in model.state_dict().items()
                    },
                    "fold": fold,
                    "epoch": epoch + 1,
                    "val_macro_f1": val_f1,
                    "model_name": model_name
                },
                checkpoint_path
            )

            print("Best validation checkpoint saved.")

        else:
            patience_count += 1

        if patience_count >= PATIENCE:
            print("Early stopping.")
            break

    # Load best validation checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    return model, best_val_f1, checkpoint["epoch"]



# MAIN EXPERIMENT

loader = FI2010Loader(str(ROOT))
results = []

print("Device:", device)
print("Starting resume-safe multi-fold experiment:", FOLDS)

for fold in FOLDS:

    print("\n" + "=" * 65)
    print("PROCESSING FOLD:", fold)
    print("=" * 65)

    X_train_raw, y_train_raw = loader.load_fold(
        fold, split="Training"
    )

    X_seq, y_seq = create_sequences(
        X_train_raw, y_train_raw, sequence_length=100
    )

    (
        X_train, y_train, X_val, y_val,
        train_indices, val_indices
    ) = chronological_split(X_seq, y_seq)

    y_train_mapped = y_train.astype(np.int64) - 1
    class_weights = calculate_class_weights(y_train_mapped)

    train_loader, val_loader = create_dataloaders(
        X_seq, y_seq, train_indices, val_indices,
        batch_size=BATCH_SIZE
    )

    X_test_raw, y_test_raw = loader.load_fold(
        fold, split="Testing"
    )

    X_test_seq, y_test_seq = create_sequences(
        X_test_raw, y_test_raw, sequence_length=100
    )

    test_dataset = FI2010TorchDataset(
        X_test_seq, y_test_seq, np.arange(len(X_test_seq))
    )

    test_loader = DataLoader(
        test_dataset, batch_size=BATCH_SIZE,
        shuffle=False, num_workers=0
    )

    for model_name, model_class in MODELS.items():

        checkpoint_path = (
            OUTPUT_DIR / f"fold{fold}_{model_name}.pt"
        )

        completion_marker = (
            OUTPUT_DIR / f"fold{fold}_{model_name}.completed"
        )

        print("\nMODEL:", model_name)

        if completion_marker.exists():

            print("Already completed. Loading saved checkpoint.")

            model = model_class().to(device)

            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=True
            )

            model.load_state_dict(checkpoint["model_state_dict"])
            best_val_f1 = checkpoint["val_macro_f1"]
            best_epoch = checkpoint["epoch"]

        else:

            print("Training model...")

            model, best_val_f1, best_epoch = train_one_model(
                model_name, model_class, fold,
                train_loader, val_loader, class_weights
            )

        test_f1 = evaluate(model, test_loader)

        print(
            f"\nFOLD {fold} | {model_name}\n"
            f"Best epoch: {best_epoch}\n"
            f"Validation Macro F1: {best_val_f1:.4f}\n"
            f"Test Macro F1: {test_f1:.4f}"
        )

        results.append({
            "fold": fold,
            "model": model_name,
            "best_epoch": best_epoch,
            "validation_macro_f1": best_val_f1,
            "test_macro_f1": test_f1
        })

        # Mark complete only after successful test evaluation.
        completion_marker.touch()

# SAVE RESULTS

csv_path = Path("results/multifold_results.csv")
csv_path.parent.mkdir(parents=True, exist_ok=True)

with open(csv_path, "w", newline="") as f:

    writer = csv.DictWriter(
        f,
        fieldnames=[
            "fold", "model", "best_epoch",
            "validation_macro_f1", "test_macro_f1"
        ]
    )

    writer.writeheader()
    writer.writerows(results)

print("\n" + "=" * 65)
print("MULTI-FOLD EXPERIMENT COMPLETED")
print("=" * 65)

for model_name in MODELS:

    scores = [
        r["test_macro_f1"]
        for r in results
        if r["model"] == model_name
    ]

    if scores:
        print(
            f"{model_name}: Mean Test Macro F1 = "
            f"{np.mean(scores):.4f}, "
            f"Std = {np.std(scores, ddof=1):.4f}"
            if len(scores) > 1
            else f"{model_name}: Test Macro F1 = {scores[0]:.4f}"
        )

print("Results saved:", csv_path)
