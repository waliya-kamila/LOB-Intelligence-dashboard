"""
Build the small FI-2010 demo slice used by the public dashboard.

The full FI-2010 test file (~90 MB) is too large for the repository, so the
deployed app falls back to a short contiguous slice of the Fold 1 test set.

FI-2010 is released under CC BY 4.0:
    Ntakaris, A., Magris, M., Kanniainen, J., Gabbouj, M., & Iosifidis, A. (2018).
    Benchmark dataset for mid-price forecasting of limit order book data with
    machine learning methods. Journal of Forecasting, 37(8), 852-866.
    https://etsin.fairdata.fi/dataset/73eb48d7-4dbc-4a10-a52a-da745b47a649

Usage:
    python -m scripts.make_demo_slice
"""

from pathlib import Path

import numpy as np

from src.data.fi2010 import FI2010Loader

ROOT = Path(__file__).resolve().parent.parent
DATASET_ROOT = ROOT / "deployment_data"
OUTPUT = ROOT / "deployment_data" / "demo" / "fi2010_cf1_test_slice.npz"

START = 0
N_OBSERVATIONS = 10_100  # gives 10,001 windows of length 100


def main():
    X, y = FI2010Loader(DATASET_ROOT).load_fold(1, split="Testing")

    X_slice = X[START:START + N_OBSERVATIONS].astype(np.float32)
    y_slice = y[START:START + N_OBSERVATIONS].astype(np.int8)

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUTPUT,
        X=X_slice,
        y=y_slice,
        start=np.int64(START),
        source=np.array("FI-2010 NoAuction ZScore Test CF_1 (CC BY 4.0)"),
    )

    print("Saved:", OUTPUT)
    print("Features:", X_slice.shape, "Labels:", y_slice.shape)
    print("Size (KB):", round(OUTPUT.stat().st_size / 1024, 1))


if __name__ == "__main__":
    main()
