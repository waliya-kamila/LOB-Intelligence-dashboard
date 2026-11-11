
import numpy as np

HORIZONS = (1, 2, 3, 5, 10)
MAX_HORIZON = max(HORIZONS)


def chronological_split(
    X,
    y,
    train_ratio=0.80,
    purge=MAX_HORIZON
):
    """
    Chronological multi-horizon train-validation split.

    X: (samples, sequence_length, features)
    y: (samples, number_of_horizons)

    Purges training samples near the boundary to
    prevent overlapping future-label horizons.
    """

    if X.ndim != 3 or y.ndim != 2:
        raise ValueError("Unexpected input dimensions.")

    if len(X) != len(y):
        raise ValueError("X and y lengths must match.")

    if not 0 < train_ratio < 1:
        raise ValueError("Invalid train ratio.")

    if purge < MAX_HORIZON:
        raise ValueError(
            f"Purge must be at least {MAX_HORIZON}."
        )

    n_samples = len(X)
    split_index = int(n_samples * train_ratio)

    train_end = split_index - purge

    if train_end <= 0 or split_index >= n_samples:
        raise ValueError("Insufficient samples.")

    # Each sequence's prediction timestamp is its final observation.
    train_indices = np.arange(train_end)
    val_indices = np.arange(split_index, n_samples)

    # Verify that training future horizons end before
    # the first validation prediction timestamp.
    last_train_endpoint = train_indices[-1]
    first_val_endpoint = val_indices[0]

    latest_train_target = (
        last_train_endpoint + MAX_HORIZON
    )

    if latest_train_target >= first_val_endpoint:
        raise ValueError("Temporal leakage boundary detected.")

    return (
        X[:train_end],
        y[:train_end],
        X[split_index:],
        y[split_index:],
        train_indices,
        val_indices
    )


if __name__ == "__main__":

    X = np.random.randn(1000, 100, 144).astype(np.float32)

    y = np.random.randint(
        1, 4, size=(1000, 5), dtype=np.int8
    )

    result = chronological_split(X, y)

    X_train, y_train, X_val, y_val, train_idx, val_idx = result

    print("\nMULTI-HORIZON LEAKAGE AUDIT")
    print("---------------------------")
    print("Horizons:", HORIZONS)
    print("Maximum horizon:", MAX_HORIZON)

    print("Training samples:", len(X_train))
    print("Validation samples:", len(X_val))

    print("Last training endpoint:", train_idx[-1])
    print("First validation endpoint:", val_idx[0])

    print(
        "Latest training target endpoint:",
        train_idx[-1] + MAX_HORIZON
    )

    print(
        "No future-target overlap:",
        train_idx[-1] + MAX_HORIZON < val_idx[0]
    )

    print("\nLeakage audit completed.")
