import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.data.fi2010 import FI2010Loader
from src.data.sequences import create_sequences
from src.data.torch_dataset import FI2010TorchDataset

from src.models.baselines import CNNGRUWithoutAttention
from src.models.multihorizon_model import TemporalCNNGRUAttention


# Configuration
FOLDS = [1, 2, 3]
HORIZONS = [1, 2, 3, 5, 10]
HORIZON_INDEX = 4

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


def main():

    loader = FI2010Loader(str(DATASET_ROOT))
    all_rows = []

    print("=" * 65)
    print("HORIZON 10 PREDICTION CONFIDENCE ANALYSIS")
    print("=" * 65)

    print("Device:", device)

    for fold in FOLDS:

        print(f"\nProcessing Fold {fold}")

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

        for model_name, model_class in MODELS.items():

            checkpoint_path = (
                CHECKPOINT_DIR
                / f"fold{fold}_{model_name}.pt"
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

            model.eval()

            with torch.no_grad():

                for X, y in test_loader:

                    X = X.to(device)

                    logits, _ = model(X)

                    probabilities = torch.softmax(
                        logits, dim=-1
                    )

                    # Select Horizon 10
                    probs = probabilities[
                        :, HORIZON_INDEX, :
                    ].cpu().numpy()

                    actual = y[
                        :, HORIZON_INDEX
                    ].numpy()

                    predicted = np.argmax(
                        probs, axis=1
                    )

                    confidence = np.max(
                        probs, axis=1
                    )

                    for i in range(len(actual)):

                        all_rows.append({
                            "fold": fold,
                            "model": model_name,
                            "actual_class": CLASS_NAMES[actual[i]],
                            "predicted_class": CLASS_NAMES[predicted[i]],
                            "prob_up": probs[i, 0],
                            "prob_stationary": probs[i, 1],
                            "prob_down": probs[i, 2],
                            "confidence": confidence[i],
                            "correct": int(
                                actual[i] == predicted[i]
                            )
                        })

            print("Completed:", model_name)

    # Save individual predictions
    df = pd.DataFrame(all_rows)

    prediction_path = (
        OUTPUT_DIR / "horizon10_prediction_probabilities.csv"
    )

    df.to_csv(prediction_path, index=False)

    # Overall confidence summary
    summary = df.groupby("model").agg(
        mean_confidence=("confidence", "mean"),
        mean_up_probability=("prob_up", "mean"),
        mean_stationary_probability=("prob_stationary", "mean"),
        mean_down_probability=("prob_down", "mean"),
        accuracy=("correct", "mean")
    ).reset_index()

    summary_path = (
        OUTPUT_DIR / "horizon10_confidence_summary.csv"
    )

    summary.to_csv(summary_path, index=False)

    print("\nOVERALL CONFIDENCE SUMMARY")
    print(summary.round(4).to_string(index=False))

    # Confidence for correct and incorrect predictions
    correctness = df.groupby(
        ["model", "correct"]
    ).agg(
        mean_confidence=("confidence", "mean"),
        sample_count=("confidence", "size")
    ).reset_index()

    correctness_path = (
        OUTPUT_DIR / "horizon10_correctness_confidence.csv"
    )

    correctness.to_csv(correctness_path, index=False)

    print("\nCONFIDENCE: CORRECT VS INCORRECT")
    print(correctness.round(4).to_string(index=False))

    # Class-wise confidence
    class_summary = df.groupby(
        ["model", "actual_class"]
    ).agg(
        mean_confidence=("confidence", "mean"),
        mean_up_probability=("prob_up", "mean"),
        mean_stationary_probability=("prob_stationary", "mean"),
        mean_down_probability=("prob_down", "mean"),
        accuracy=("correct", "mean"),
        sample_count=("correct", "size")
    ).reset_index()

    class_path = (
        OUTPUT_DIR / "horizon10_class_confidence.csv"
    )

    class_summary.to_csv(class_path, index=False)

    print("\nCLASS-WISE CONFIDENCE")
    print(class_summary.round(4).to_string(index=False))

    print("\nSaved files:")
    print(prediction_path)
    print(summary_path)
    print(correctness_path)
    print(class_path)


if __name__ == "__main__":
    main()
