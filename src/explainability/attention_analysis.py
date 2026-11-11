
from pathlib import Path
import numpy as np
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from src.data.fi2010 import FI2010Loader
from src.data.sequences import create_sequences
from src.data.torch_dataset import FI2010TorchDataset

from src.models.multihorizon_model import TemporalCNNGRUAttention


# ==========================================
# CONFIGURATION
# ==========================================

FOLD = 1
BATCH_SIZE = 8
SEQUENCE_LENGTH = 100

DATASET_ROOT = (
    Path.home()
    / "Downloads"
    / "BenchmarkDatasets"
    / "NoAuction"
    / "1.NoAuction_Zscore"
)

CHECKPOINT_PATH = Path(
    "results/checkpoints/multifold/"
    "fold1_CNN-GRU-Attention.pt"
)

OUTPUT_DIR = Path("results/explainability")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device(
    "mps" if torch.backends.mps.is_available() else "cpu"
)


# ==========================================
# LOAD TEST DATA
# ==========================================

print("Loading FI-2010 test data...")

loader = FI2010Loader(str(DATASET_ROOT))

X_raw, y_raw = loader.load_fold(
    FOLD,
    split="Testing"
)

X_seq, y_seq = create_sequences(
    X_raw,
    y_raw,
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


# ==========================================
# LOAD TRAINED MODEL
# ==========================================

model = TemporalCNNGRUAttention().to(device)

checkpoint = torch.load(
    CHECKPOINT_PATH,
    map_location="cpu",
    weights_only=True
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.eval()

print("Checkpoint epoch:", checkpoint["epoch"])


# ==========================================
# EXTRACT ATTENTION WEIGHTS
# ==========================================

all_attention = []
all_predictions = []
all_targets = []

with torch.no_grad():

    for X, y in test_loader:

        X = X.to(device)

        logits, attention = model(X)

        predictions = logits.argmax(dim=-1)

        all_attention.append(
            attention.cpu().numpy()
        )

        all_predictions.append(
            predictions.cpu().numpy()
        )

        all_targets.append(
            y.numpy()
        )


attention_weights = np.concatenate(
    all_attention,
    axis=0
)

predictions = np.concatenate(
    all_predictions,
    axis=0
)

targets = np.concatenate(
    all_targets,
    axis=0
)


# ==========================================
# VALIDATE ATTENTION
# ==========================================

print("\nAttention shape:", attention_weights.shape)

print(
    "Attention sum for first sample:",
    attention_weights[0].sum()
)

assert attention_weights.shape[1] == SEQUENCE_LENGTH

assert np.allclose(
    attention_weights.sum(axis=1),
    1.0,
    atol=1e-5
)

print("Attention validation passed.")

# ATTENTION CONCENTRATION DIAGNOSTICS

epsilon = 1e-12

max_attention = attention_weights.max(axis=1)

max_positions = attention_weights.argmax(axis=1)

last_1 = attention_weights[:, -1].mean()

last_5 = attention_weights[:, -5:].sum(axis=1).mean()

last_10 = attention_weights[:, -10:].sum(axis=1).mean()

entropy = -np.sum(
    attention_weights *
    np.log(attention_weights + epsilon),
    axis=1
)

normalized_entropy = entropy / np.log(100)

print("\n========== ATTENTION DIAGNOSTICS ==========")

print("Mean maximum attention:", max_attention.mean())

print("Mean attention at last step:", last_1)

print("Mean attention in last 5 steps:", last_5)

print("Mean attention in last 10 steps:", last_10)

print(
    "Percentage with maximum at final step:",
    np.mean(max_positions == 99) * 100
)

print("Mean normalized entropy:", normalized_entropy.mean())

print(
    "Mean attention in first 90 steps:",
    attention_weights[:, :90].sum(axis=1).mean()
)



# ==========================================
# SAVE ATTENTION VALUES
# ==========================================

np.save(
    OUTPUT_DIR / "fold1_attention_weights.npy",
    attention_weights
)


# ==========================================
# AVERAGE ATTENTION DISTRIBUTION
# ==========================================

mean_attention = attention_weights.mean(axis=0)

np.save(
    OUTPUT_DIR / "fold1_mean_attention.npy",
    mean_attention
)


# ==========================================
# PLOT MEAN ATTENTION
# ==========================================

plt.figure(figsize=(12, 5))

plt.plot(
    np.arange(1, SEQUENCE_LENGTH + 1),
    mean_attention
)

plt.xlabel("Time Step")
plt.ylabel("Mean Attention Weight")

plt.title(
    "Mean Temporal Attention Distribution - Fold 1"
)

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "fold1_mean_attention.png",
    dpi=300
)

plt.close()


# ==========================================
# ATTENTION HEATMAP
# ==========================================

plt.figure(figsize=(14, 6))

plt.imshow(
    attention_weights[:50],
    aspect="auto",
    interpolation="nearest"
)

plt.xlabel("Time Step")
plt.ylabel("Test Sample")

plt.title(
    "Temporal Attention Heatmap - First 50 Samples"
)

plt.colorbar(
    label="Attention Weight"
)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "fold1_attention_heatmap.png",
    dpi=300
)

plt.close()


# ==========================================
# FINAL OUTPUT
# ==========================================

print("\nEXPLAINABILITY EXPERIMENT COMPLETED")

print("Mean attention:", mean_attention.shape)

print("Files saved in:", OUTPUT_DIR)
