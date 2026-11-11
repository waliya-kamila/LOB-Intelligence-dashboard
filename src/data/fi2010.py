
from pathlib import Path

import numpy as np


# Dataset configuration
N_FEATURES = 144
N_LABELS = 5
N_CLASSES = 3

# FI-2010 label encoding
LABEL_NAMES = {
    1: "Up",
    2: "Stationary",
    3: "Down",
}

# Common FI-2010 horizon ordering.
# Keep this explicit in experiment documentation.
HORIZONS = (1, 2, 3, 5, 10)


class FI2010Loader:
    """
    Memory-conscious loader for one FI-2010 fold.

    Raw file layout:
        Rows 1-144: features
        Rows 145-149: labels

    Returned layout:
        X: (observations, 144)
        y: (observations, 5)
    """

    def __init__(self, dataset_root):
        self.root = Path(dataset_root).expanduser()

        if not self.root.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.root}"
            )

    def get_file(self, fold, split="Training"):
        if fold not in range(1, 10):
            raise ValueError("Fold must be between 1 and 9.")

        if split not in ("Training", "Testing"):
            raise ValueError(
                "Split must be 'Training' or 'Testing'."
            )

        folder_name = f"NoAuction_Zscore_{split}"
        prefix = "Train" if split == "Training" else "Test"

        filename = (
            f"{prefix}_Dst_NoAuction_ZScore_CF_{fold}.txt"
        )

        path = (
            self.root
            / folder_name
            / filename
        )

        if not path.is_file():
            raise FileNotFoundError(path)

        return path

    def load_fold(self, fold, split="Training"):
        path = self.get_file(fold, split)

        feature_rows = []
        label_rows = []

        n_observations = None

        with path.open("r") as file:
            for row_number, line in enumerate(file, start=1):

                values = np.fromstring(line, sep=" ")

                if n_observations is None:
                    n_observations = len(values)

                    if n_observations == 0:
                        raise ValueError("Empty dataset row.")

                if len(values) != n_observations:
                    raise ValueError(
                        f"Inconsistent columns at row {row_number}"
                    )

                if not np.isfinite(values).all():
                    raise ValueError(
                        f"Non-finite values at row {row_number}"
                    )

                if row_number <= N_FEATURES:
                    feature_rows.append(
                        values.astype(np.float32)
                    )
                

                elif row_number <= N_FEATURES + N_LABELS:
                    labels = values.astype(np.int8)

                    if not np.isin(
                        labels, [1, 2, 3]
                    ).all():
                        raise ValueError(
                            f"Invalid labels at row {row_number}"
                        )

                    label_rows.append(labels)

                else:
                    raise ValueError(
                        f"Unexpected extra row: {row_number}"
                    )

        if len(feature_rows) != N_FEATURES:
            raise ValueError(
                f"Expected 144 feature rows, got {len(feature_rows)}"
            )

        if len(label_rows) != N_LABELS:
            raise ValueError(
                f"Expected 5 label rows, got {len(label_rows)}"
            )

        # Convert raw row-oriented data into sample-oriented arrays.
        X = np.stack(feature_rows, axis=0).T
        y = np.stack(label_rows, axis=0).T

        if X.shape != (n_observations, N_FEATURES):
            raise ValueError("Unexpected feature matrix shape.")

        if y.shape != (n_observations, N_LABELS):
            raise ValueError("Unexpected label matrix shape.")

        return X, y


if __name__ == "__main__":

    dataset_root = (
        "~/Downloads/BenchmarkDatasets/"
        "NoAuction/1.NoAuction_Zscore"
    )

    loader = FI2010Loader(dataset_root)

    X, y = loader.load_fold(
        fold=1,
        split="Training"
    )

    print("\nFI-2010 LOADER VALIDATION")
    print("-------------------------")
    print("Feature shape:", X.shape)
    print("Label shape:", y.shape)
    print("Feature dtype:", X.dtype)
    print("Label dtype:", y.dtype)
    print("All features finite:", np.isfinite(X).all())
    print("All labels valid:", np.isin(y, [1, 2, 3]).all())
    print("Label names:", LABEL_NAMES)
    print("Horizons:", HORIZONS)
    print("\nLoader validation completed.")
