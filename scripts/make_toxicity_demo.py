"""
Build the Stage 2 demo bundle used by the dashboard's Toxicity Lab page.

Writes
    deployment_data/demo/toxicity_btcusdt_2024-10-01.npz
        X        (T, 49) float32  model inputs, already normalised exactly as in training
        y        (T,)    int8     toxic label (1 = toxic) for each second
        lp_loss  (T,)    float32  realised liquidity-provider loss over the next 30 s (bps x BTC)
        vpin     (T,)    float32  VPIN available at each second
        mid      (T,)    float32  BTCUSDT mid-price
        start    int              first second of the slice (seconds after 00:00 UTC)
    results/checkpoints/toxicity_foldB_lob_trades.pt
        Fold B toxicity detector (trained on 1 Mar - 1 Jul 2024, tested on 1 Oct 2024)
        saved with tensors only so it loads with torch.load(weights_only=True).

The slice covers 20:00-22:00 UTC on the Fold B test day, which the model never saw in training.
Only derived, normalised features are stored, not raw exchange data.

Usage:
    TOX_LABEL=exposure python -m scripts.make_toxicity_demo
"""

import os
from pathlib import Path

os.environ.setdefault("TOX_LABEL", "exposure")

import numpy as np
import torch

from src.toxicity import experiment as E

ROOT = Path(__file__).resolve().parent.parent
OUT_DATA = ROOT / "deployment_data" / "demo" / "toxicity_btcusdt_2024-10-01.npz"
OUT_CKPT = ROOT / "results" / "checkpoints" / "toxicity_foldB_lob_trades.pt"
START, END = 20 * 3600 - E.WINDOW, 22 * 3600   # include 100 s of history before 20:00


def main():
    train_ids, val_id, test_id = E.FOLDS["B"]
    days = E.load_days()
    E.prepare(days, train_ids)
    d = days[test_id]

    ck = torch.load(ROOT / "results" / "toxicity_exposure" / "foldB_lob_trades.pt",
                    map_location="cpu", weights_only=False)
    idx = ck["feat_idx"]
    X = np.clip((d["X_full"][:, idx] - ck["mean"]) / ck["std"], -10, 10).astype(np.float32)

    sl = slice(START, END)
    OUT_DATA.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT_DATA,
        X=X[sl], y=d["y"][sl].astype(np.int8),
        lp_loss=np.nan_to_num(d["tox"][sl]).astype(np.float32),
        vpin=np.nan_to_num(d["vpin"][sl]).astype(np.float32),
        mid=d["mid"][sl].astype(np.float32), start=np.int64(START),
    )
    torch.save({
        "model_state_dict": ck["model_state_dict"],
        "input_features": len(idx),
        "epoch": int(ck["epoch"]),
        "val_auc": float(ck["val_auc"]),
        "train_days": [E.DAYS[i] for i in train_ids],
        "test_day": E.DAYS[test_id],
    }, OUT_CKPT)

    # sanity check: the clean checkpoint reproduces the saved test scores
    model = E.TemporalCNNGRUAttention(input_features=len(idx), num_horizons=1, num_classes=2, dropout=E.DROPOUT)
    model.load_state_dict(torch.load(OUT_CKPT, map_location="cpu", weights_only=True)["model_state_dict"])
    model.eval()
    ref = np.load(ROOT / "results" / "toxicity_exposure" / "foldB_lob_trades_test_scores.npz")
    t_check = [t for t in ref["t"] if 20 * 3600 <= t < 22 * 3600][:200]
    with torch.no_grad():
        w = torch.from_numpy(np.stack([X[t - E.WINDOW + 1:t + 1] for t in t_check]))
        p = torch.softmax(model(w)[0][:, 0, :], -1)[:, 1].numpy()
    lookup = dict(zip(ref["t"].tolist(), ref["score"].tolist()))
    err = max(abs(p[k] - lookup[t]) for k, t in enumerate(t_check))
    print(f"saved {OUT_DATA.name} ({OUT_DATA.stat().st_size / 1e6:.2f} MB), {OUT_CKPT.name}; "
          f"max score difference vs experiment: {err:.2e}; toxic rate in slice: {d['y'][20*3600:END].mean():.3f}")


if __name__ == "__main__":
    main()
