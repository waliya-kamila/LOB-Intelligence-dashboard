import numpy as np
import torch


LABEL_NAMES = {
    0: "Up",
    1: "Stationary",
    2: "Down"
}


def calculate_class_weights(y_train):
    """
    Calculate balanced class weights independently
    for each prediction horizon.

    y_train shape: (N, 5)
    """

    if y_train.ndim != 2 or y_train.shape[1] != 5:
        raise ValueError("Expected y_train shape (N, 5)")

    weights = []

    for horizon in range(y_train.shape[1]):

        labels = y_train[:, horizon].astype(np.int64)

        if not np.isin(labels, [0, 1, 2]).all():
            raise ValueError("Labels must be mapped to 0, 1, 2")

        counts = np.bincount(labels, minlength=3)

        if np.any(counts == 0):
            raise ValueError(
                f"Missing class in horizon {horizon + 1}"
            )

        total = len(labels)

        horizon_weights = total / (3.0 * counts)

        weights.append(horizon_weights)

    weights = np.asarray(weights, dtype=np.float32)

    return torch.tensor(weights, dtype=torch.float32)


def print_class_distribution(y_train):

    print("\nCLASS DISTRIBUTION BY HORIZON")
    print("-" * 55)

    for horizon in range(y_train.shape[1]):

        labels = y_train[:, horizon].astype(np.int64)

        counts = np.bincount(labels, minlength=3)

        total = len(labels)

        print(f"\nHorizon {horizon + 1}")

        for class_id, class_name in LABEL_NAMES.items():

            percentage = counts[class_id] / total * 100

            print(
                f"{class_name:12s}: "
                f"{counts[class_id]:6d} "
                f"({percentage:.2f}%)"
            )