
import numpy as np


def create_sequences(X, y, sequence_length=100):
    """
    Generate chronological sliding-window sequences.

    Parameters
    ----------
    X : ndarray
        Features with shape (observations, features).

    y : ndarray
        Multi-horizon labels with shape (observations, 5).

    sequence_length : int
        Number of consecutive observations per sequence.

    Returns
    -------
    X_seq : ndarray
        Shape (samples, sequence_length, features).

    y_seq : ndarray
        Shape (samples, 5).
    """

    if X.ndim != 2:
        raise ValueError("X must be a 2D array.")

    if y.ndim != 2:
        raise ValueError("y must be a 2D array.")

    if len(X) != len(y):
        raise ValueError("X and y must have equal observations.")

    if sequence_length <= 0:
        raise ValueError("Sequence length must be positive.")

    if sequence_length > len(X):
        raise ValueError("Sequence length exceeds observations.")

    n_samples = len(X) - sequence_length + 1

    X_seq = np.lib.stride_tricks.sliding_window_view(
        X,
        window_shape=sequence_length,
        axis=0
    )

    X_seq = np.moveaxis(X_seq, -1, 1)

    # Target corresponds to the final observation in each window.
    y_seq = y[sequence_length - 1:]

    assert X_seq.shape == (
        n_samples,
        sequence_length,
        X.shape[1]
    )

    assert y_seq.shape == (
        n_samples,
        y.shape[1]
    )

    return X_seq, y_seq


if __name__ == "__main__":

    # Small synthetic test; does not load the full dataset.
    X = np.random.randn(200, 144).astype(np.float32)

    y = np.random.randint(
        1, 4, size=(200, 5), dtype=np.int8
    )

    X_seq, y_seq = create_sequences(
        X,
        y,
        sequence_length=100
    )

    print("\nSEQUENCE GENERATION VALIDATION")
    print("--------------------------------")
    print("Original features:", X.shape)
    print("Original labels:", y.shape)
    print("Sequence features:", X_seq.shape)
    print("Sequence labels:", y_seq.shape)
    print("Feature dtype:", X_seq.dtype)
    print("Label dtype:", y_seq.dtype)

    print(
        "First target alignment:",
        np.array_equal(y_seq[0], y[99])
    )

    print(
        "Last target alignment:",
        np.array_equal(y_seq[-1], y[-1])
    )

    print("\nSequence generation completed.")
