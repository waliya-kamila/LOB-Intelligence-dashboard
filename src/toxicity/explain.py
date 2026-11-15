"""
Explain the toxicity detector with Integrated Gradients and attention.

For the Fold B model (LOB + trades), Integrated Gradients (64 steps, baseline =
training mean, i.e. zero after standardisation) is computed for the toxic-class
output on the test-day windows with the highest toxicity scores that are truly
toxic, plus an equal number of random non-toxic windows.

Outputs (results/toxicity):
    ig_feature_importance.csv   mean |attribution| per input feature
    ig_group_importance.csv     share by feature group
    ig_temporal_profile.csv     share of attribution by time step (100 = now)
    attention_profile.csv       mean attention weight by time step

Usage:
    python -m src.toxicity.explain
"""

import csv
from pathlib import Path

import numpy as np
import torch

from src.models.multihorizon_model import TemporalCNNGRUAttention
from src.toxicity import labels as L
from src.toxicity.experiment import DAYS, FOLDS, WINDOW, load_days, prepare

ROOT = Path(__file__).resolve().parents[2]
from src.toxicity.experiment import OUT
N_PER_CLASS = 150
STEPS = 64
DEVICE = torch.device("cpu")   # IG is cheap; CPU keeps gradients deterministic

GROUPS = {
    "Ask prices (distance)": range(0, 10), "Ask volumes": range(10, 20),
    "Bid prices (distance)": range(20, 30), "Bid volumes": range(30, 40),
    "Mid-price return": range(40, 41), "Trade flow": range(41, 44),
    "Flow context (60 s)": range(44, 49),
}


def main():
    train_ids, val_id, test_id = FOLDS["B"]
    days = load_days()
    prepare(days, train_ids)
    ck = torch.load(OUT / "foldB_lob_trades.pt", map_location="cpu", weights_only=False)
    mean, std = ck["mean"], ck["std"]
    idx = ck["feat_idx"]
    model = TemporalCNNGRUAttention(input_features=len(idx), num_horizons=1, num_classes=2, dropout=0.4)
    model.load_state_dict(ck["model_state_dict"]); model.eval()

    d = days[test_id]
    X = np.clip((d["X_full"][:, idx] - mean) / std, -10, 10).astype(np.float32)
    sc = np.load(OUT / "foldB_lob_trades_test_scores.npz")
    t_idx, y, s = sc["t"], sc["y"], sc["score"]
    rng = np.random.default_rng(42)
    pos = t_idx[y == 1][np.argsort(-s[y == 1])[:N_PER_CLASS]]
    neg = rng.choice(t_idx[y == 0], N_PER_CLASS, replace=False)
    sel = np.concatenate([pos, neg])

    windows = torch.from_numpy(np.stack([X[t - WINDOW + 1:t + 1] for t in sel]))
    base = torch.zeros_like(windows[:1])
    total = torch.zeros_like(windows)
    for k in range(1, STEPS + 1):
        a = k / STEPS
        x = (base + a * (windows - base)).requires_grad_(True)
        logits, _ = model(x)
        logits[:, 0, 1].sum().backward()
        total += x.grad
    attr = (windows - base) * total / STEPS
    with torch.no_grad():
        f_in = model(windows)[0][:, 0, 1]; f_b = model(base)[0][:, 0, 1]
        _, att = model(windows)
    completeness = (attr.sum(dim=(1, 2)) - (f_in - f_b)).abs().mean().item()
    print("IG mean completeness error:", round(completeness, 5))

    A = attr.abs().numpy()
    toxic = A[:N_PER_CLASS]
    feat = toxic.sum(axis=1).mean(axis=0)
    share = feat / feat.sum()
    with open(OUT / "ig_feature_importance.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["feature", "share"])
        for i in np.argsort(-share):
            w.writerow([L.FEATURE_NAMES[i], round(float(share[i]), 5)])
    with open(OUT / "ig_group_importance.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["group", "share_toxic", "share_nontoxic"])
        nf = A[N_PER_CLASS:].sum(axis=1).mean(axis=0); nf = nf / nf.sum()
        for g, idx in GROUPS.items():
            w.writerow([g, round(float(share[list(idx)].sum()), 4), round(float(nf[list(idx)].sum()), 4)])
    temporal = toxic.sum(axis=2).mean(axis=0); temporal = temporal / temporal.sum()
    att_mean = att.numpy().mean(axis=0)
    with open(OUT / "ig_temporal_profile.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["step", "ig_share", "attention"])
        for i in range(WINDOW):
            w.writerow([i + 1, round(float(temporal[i]), 6), round(float(att_mean[i]), 6)])
    level = [float(share[[i, 10 + i, 20 + i, 30 + i]].sum()) for i in range(10)]
    print("Group shares (toxic):", {g: round(float(share[list(i)].sum()), 3) for g, i in GROUPS.items()})
    print("Level shares L1..L10:", [round(v, 3) for v in level])
    print("IG share last 10 s:", round(float(temporal[-10:].sum()), 3), " attention last step:", round(float(att_mean[-1]), 3))
    print("Top features:", [L.FEATURE_NAMES[i] for i in np.argsort(-share)[:10]])


if __name__ == "__main__":
    main()
