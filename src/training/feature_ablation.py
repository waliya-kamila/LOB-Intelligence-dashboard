
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
from src.data.torch_dataset import (
    create_dataloaders,
    FI2010TorchDataset,
)

from src.models.baselines import CNNGRUWithoutAttention
from src.models.multihorizon_model import TemporalCNNGRUAttention
from src.training.losses import MultiHorizonLoss


# ============================================================
# CONFIGURATION
# ============================================================

FOLDS = [1, 2, 3]
SEQUENCE_LENGTH = 100

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

CHECKPOINT_ROOT = Path(
    "results/checkpoints/feature_ablation"
)

RESULTS_DIR = Path(
    "results/feature_ablation"
)

CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)

MODEL_CLASSES = {
    "CNN-GRU-NoAttention": CNNGRUWithoutAttention,
    "CNN-GRU-Attention": TemporalCNNGRUAttention,
}


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# ============================================================
# METRIC
# ============================================================

def macro_f1(y_true, y_pred):

    scores = []

    for horizon in range(5):

        class_scores = []

        for class_id in range(3):

            tp = np.sum(
                (y_true[:, horizon] == class_id)
                & (y_pred[:, horizon] == class_id)
            )

            fp = np.sum(
                (y_true[:, horizon] != class_id)
                & (y_pred[:, horizon] == class_id)
            )

            fn = np.sum(
                (y_true[:, horizon] == class_id)
                & (y_pred[:, horizon] != class_id)
            )

            precision = (
                tp / (tp + fp) if tp + fp else 0
            )

            recall = (
                tp / (tp + fn) if tp + fn else 0
            )

            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall else 0
            )

            class_scores.append(f1)

        scores.append(np.mean(class_scores))

    return float(np.mean(scores))


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader):

    model.eval()

    all_predictions = []
    all_targets = []

    with torch.no_grad():

        for X, y in loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            logits, _ = model(X)

            predictions = logits.argmax(dim=-1)

            all_predictions.append(
                predictions.cpu().numpy()
            )

            all_targets.append(
                y.cpu().numpy()
            )

    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)

    return macro_f1(targets, predictions)


# ============================================================
# TRAINING-ONLY FEATURE AUDIT
# ============================================================

def get_training_feature_mask(
    X_raw,
    train_indices,
    sequence_length,
):
    """
    Identify constant features using only raw observations
    appearing in training input sequences.

    Validation and test observations are excluded.
    """

    train_indices = np.asarray(
        train_indices,
        dtype=np.int64
    )

    if len(train_indices) == 0:
        raise ValueError("No training indices found.")

    expected = np.arange(
        train_indices.min(),
        train_indices.max() + 1
    )

    if not np.array_equal(train_indices, expected):
        raise ValueError(
            "Training sequence indices are not contiguous. "
            "Review chronological_split before feature selection."
        )

    start = int(train_indices.min())

    end = int(
        train_indices.max() + sequence_length
    )

    training_observations = X_raw[start:end]

    if len(training_observations) == 0:
        raise ValueError("Empty training observation range.")

    feature_std = np.std(
        training_observations,
        axis=0
    )

    constant_mask = feature_std == 0

    return constant_mask, feature_std


# ============================================================
# TRAIN ONE MODEL
# ============================================================

def train_one_model(
    model_name,
    model_class,
    input_features,
    fold,
    train_loader,
    val_loader,
    class_weights,
    checkpoint_path,
):

    set_seed(42 + fold)

    model = model_class(
        input_features=input_features
    ).to(DEVICE)

    criterion = MultiHorizonLoss(
        class_weights
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    best_val_f1 = -1.0
    best_epoch = 0
    patience_count = 0

    for epoch in range(EPOCHS):

        model.train()

        for X, y in train_loader:

            X = X.to(DEVICE)
            y = y.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)

            logits, _ = model(X)

            loss = criterion(logits, y)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0,
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
            best_epoch = epoch + 1
            patience_count = 0

            torch.save(
                {
                    "model_state_dict": {
                        key: value.detach().cpu()
                        for key, value in model.state_dict().items()
                    },
                    "fold": fold,
                    "epoch": best_epoch,
                    "val_macro_f1": best_val_f1,
                    "model_name": model_name,
                    "input_features": input_features,
                },
                checkpoint_path,
            )

        else:

            patience_count += 1

        if patience_count >= PATIENCE:

            print("Early stopping.")
            break

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    return model, best_val_f1, best_epoch


# ============================================================
# PREPARE FOLDS AND FEATURE MASKS
# ============================================================

loader = FI2010Loader(str(ROOT))

fold_data = {}
fold_constant_masks = []
audit_rows = []

print("Device:", DEVICE)
print("\nPHASE 1: TRAINING-ONLY FEATURE AUDIT")

for fold in FOLDS:

    X_raw, y_raw = loader.load_fold(
        fold,
        split="Training",
    )

    X_seq, y_seq = create_sequences(
        X_raw,
        y_raw,
        sequence_length=SEQUENCE_LENGTH,
    )

    (
        X_train,
        y_train,
        X_val,
        y_val,
        train_indices,
        val_indices,
    ) = chronological_split(X_seq, y_seq)

    constant_mask, feature_std = get_training_feature_mask(
        X_raw,
        train_indices,
        SEQUENCE_LENGTH,
    )

    fold_constant_masks.append(constant_mask)

    fold_data[fold] = {
        "X_raw": X_raw,
        "y_raw": y_raw,
        "X_seq": X_seq,
        "y_seq": y_seq,
        "train_indices": train_indices,
        "val_indices": val_indices,
        "y_train": y_train,
    }

    constant_indices = np.where(constant_mask)[0]

    print(
        f"\nFold {fold}: "
        f"{len(constant_indices)} constant features"
    )

    print(
        "Zero-based feature indices:",
        constant_indices.tolist(),
    )

    for feature_idx in range(X_raw.shape[1]):

        audit_rows.append({
            "fold": fold,
            "feature_index": feature_idx,
            "dataset_row": feature_idx + 1,
            "training_std": float(feature_std[feature_idx]),
            "is_constant_in_training": bool(
                constant_mask[feature_idx]
            ),
        })


# Only features constant in every training fold are removed.
common_constant_mask = np.logical_and.reduce(
    fold_constant_masks
)

keep_mask = ~common_constant_mask

constant_indices = np.where(
    common_constant_mask
)[0]

keep_indices = np.where(keep_mask)[0]

print("\n" + "=" * 65)
print("COMMON TRAINING-ONLY FEATURE AUDIT")
print("=" * 65)

print(
    "Common constant features:",
    len(constant_indices),
)

print(
    "Removed zero-based indices:",
    constant_indices.tolist(),
)

print(
    "Retained features:",
    len(keep_indices),
)

print(
    "Reduced feature count:",
    len(keep_indices),
)

with open(
    RESULTS_DIR / "training_feature_audit.csv",
    "w",
    newline="",
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "fold",
            "feature_index",
            "dataset_row",
            "training_std",
            "is_constant_in_training",
        ],
    )

    writer.writeheader()
    writer.writerows(audit_rows)


# ============================================================
# MAIN ABLATION EXPERIMENT
# ============================================================

results = []

EXPERIMENTS = {
    "All144": None,
    "Reduced": keep_indices,
}

print("\nPHASE 2: FEATURE ABLATION TRAINING")

for fold in FOLDS:

    print("\n" + "=" * 65)
    print("PROCESSING FOLD:", fold)
    print("=" * 65)

    data = fold_data[fold]

    X_seq = data["X_seq"]
    y_seq = data["y_seq"]

    train_indices = data["train_indices"]
    val_indices = data["val_indices"]

    y_train = data["y_train"]

    y_train_mapped = (
        y_train.astype(np.int64) - 1
    )

    class_weights = calculate_class_weights(
        y_train_mapped
    )

    X_test_raw, y_test_raw = loader.load_fold(
        fold,
        split="Testing",
    )

    X_test_seq, y_test_seq = create_sequences(
        X_test_raw,
        y_test_raw,
        sequence_length=SEQUENCE_LENGTH,
    )

    for experiment_name, feature_indices in EXPERIMENTS.items():

        if feature_indices is None:

            X_current = X_seq
            X_test_current = X_test_seq

        else:

            X_current = X_seq[:, :, feature_indices]
            X_test_current = X_test_seq[:, :, feature_indices]

        input_features = X_current.shape[-1]

        print(
            f"\nEXPERIMENT: {experiment_name} | "
            f"Input features: {input_features}"
        )

        train_loader, val_loader = create_dataloaders(
            X_current,
            y_seq,
            train_indices,
            val_indices,
            batch_size=BATCH_SIZE,
        )

        test_dataset = FI2010TorchDataset(
            X_test_current,
            y_test_seq,
            np.arange(len(X_test_current)),
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=0,
        )

        experiment_checkpoint_dir = (
            CHECKPOINT_ROOT / experiment_name
        )

        experiment_checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        for model_name, model_class in MODEL_CLASSES.items():

            checkpoint_path = (
                experiment_checkpoint_dir
                / f"fold{fold}_{model_name}.pt"
            )

            completion_marker = (
                experiment_checkpoint_dir
                / f"fold{fold}_{model_name}.completed"
            )

            print("\nMODEL:", model_name)

            if completion_marker.exists():

                if not checkpoint_path.exists():
                    raise FileNotFoundError(
                        f"Completion marker exists but checkpoint "
                        f"is missing: {checkpoint_path}"
                    )

                print("Loading completed experiment checkpoint.")

                model = model_class(
                    input_features=input_features
                ).to(DEVICE)

                checkpoint = torch.load(
                    checkpoint_path,
                    map_location="cpu",
                    weights_only=True,
                )

                model.load_state_dict(
                    checkpoint["model_state_dict"]
                )

                best_val_f1 = checkpoint["val_macro_f1"]
                best_epoch = checkpoint["epoch"]

            else:

                print("Training model...")

                model, best_val_f1, best_epoch = train_one_model(
                    model_name,
                    model_class,
                    input_features,
                    fold,
                    train_loader,
                    val_loader,
                    class_weights,
                    checkpoint_path,
                )

            test_f1 = evaluate(
                model,
                test_loader,
            )

            print(
                f"\nFold: {fold}\n"
                f"Experiment: {experiment_name}\n"
                f"Model: {model_name}\n"
                f"Features: {input_features}\n"
                f"Best epoch: {best_epoch}\n"
                f"Validation Macro F1: {best_val_f1:.4f}\n"
                f"Test Macro F1: {test_f1:.4f}"
            )

            results.append({
                "fold": fold,
                "experiment": experiment_name,
                "model": model_name,
                "input_features": input_features,
                "best_epoch": best_epoch,
                "validation_macro_f1": best_val_f1,
                "test_macro_f1": test_f1,
            })

            completion_marker.touch()


# ============================================================
# SAVE RESULTS
# ============================================================

results_path = (
    RESULTS_DIR / "feature_ablation_results.csv"
)

with open(
    results_path,
    "w",
    newline="",
) as file:

    fieldnames = [
        "fold",
        "experiment",
        "model",
        "input_features",
        "best_epoch",
        "validation_macro_f1",
        "test_macro_f1",
    ]

    writer = csv.DictWriter(
        file,
        fieldnames=fieldnames,
    )

    writer.writeheader()
    writer.writerows(results)


print("\n" + "=" * 65)
print("FEATURE ABLATION COMPLETED")
print("=" * 65)

for experiment_name in EXPERIMENTS:

    for model_name in MODEL_CLASSES:

        scores = [
            row["test_macro_f1"]
            for row in results
            if row["experiment"] == experiment_name
            and row["model"] == model_name
        ]

        if scores:

            print(
                f"{experiment_name} | {model_name} | "
                f"Mean Test Macro F1: {np.mean(scores):.4f} | "
                f"SD: {np.std(scores, ddof=1):.4f}"
                if len(scores) > 1
                else
                f"{experiment_name} | {model_name} | "
                f"Test Macro F1: {scores[0]:.4f}"
            )

print("\nFeature audit:", RESULTS_DIR / "training_feature_audit.csv")
print("Experiment results:", results_path)
print("Checkpoints:", CHECKPOINT_ROOT)
