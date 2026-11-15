"""
Early detection of toxic order flow: deep model vs VPIN and baselines.

Walk-forward folds over six trading days (Binance BTCUSDT futures, 2024):
    Fold A: train Mar-01, Apr-01, May-01 | validate Jul-01 | test Aug-01
    Fold B: train Mar-01 .. Jul-01        | validate Aug-01 | test Oct-01

Models
    CNN-BiGRU-Attention (LOB + trade flow, 44 features)   proposed
    CNN-BiGRU-Attention (LOB only, 41 features)           full-book ablation
    Logistic regression on 11 hand-crafted features        ML baseline
    VPIN                                                   traditional toxicity metric
    Past realised toxicity                                 persistence baseline

Usage:
    python -m src.toxicity.experiment
"""

import csv
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.models.multihorizon_model import TemporalCNNGRUAttention
from src.toxicity import labels as L

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_crypto" / "processed"
LABEL = os.environ.get("TOX_LABEL", "intensity")      # "intensity" or "exposure"
OUT = ROOT / "results" / ("toxicity" if LABEL == "intensity" else f"toxicity_{LABEL}")

DAYS = ["2024-03-01", "2024-04-01", "2024-05-01", "2024-07-01", "2024-08-01", "2024-10-01"]
FOLDS = {"A": ([0, 1, 2], 3, 4), "B": ([0, 1, 2, 3], 4, 5)}
WINDOW = 100
TOXIC_QUANTILE = 0.80
SEED = 42
EPOCHS = 12
PATIENCE = 3
BATCH = 256
LR = float(os.environ.get("TOX_LR", 3e-4))
WD = float(os.environ.get("TOX_WD", 1e-2))
TRAIN_STRIDE = int(os.environ.get("TOX_STRIDE", 5))
VAL_STRIDE = 5
DROPOUT = float(os.environ.get("TOX_DROPOUT", 0.4))
# feature indices: 0-39 book, 40 mid return, 41-43 trade flow, 44-48 flow context
FEATS = {"lob_trades": list(range(49)), "lob_only": list(range(41)) + [47]}   # LOB-only keeps spread (47), a book quantity

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")


def set_seed(seed=SEED):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


# ------------------------------------------------------------------ metrics
def roc_auc(y, s):
    order = np.argsort(s, kind="mergesort")
    ss = s[order]
    _, inv, cnt = np.unique(ss, return_inverse=True, return_counts=True)
    c = np.cumsum(cnt); avg = (c - cnt + 1 + c) / 2.0
    ranks = np.empty(len(s)); ranks[order] = avg[inv]
    n1 = y.sum(); n0 = len(y) - n1
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def average_precision(y, s):
    order = np.argsort(-s, kind="mergesort")
    yy = y[order]
    tp = np.cumsum(yy); k = np.arange(1, len(yy) + 1)
    return float((tp / k)[yy == 1].sum() / max(yy.sum(), 1))


def best_f1_threshold(y, s):
    order = np.argsort(-s, kind="mergesort")
    yy = y[order]; ss = s[order]
    tp = np.cumsum(yy); fp = np.cumsum(1 - yy); fn = yy.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    i = int(np.argmax(f1))
    return float(ss[i])


def threshold_metrics(y, s, thr):
    p = s >= thr
    tp = int((p & (y == 1)).sum()); fp = int((p & (y == 0)).sum()); fn = int((~p & (y == 1)).sum())
    prec = tp / max(tp + fp, 1); rec = tp / max(tp + fn, 1)
    return prec, rec, 2 * prec * rec / max(prec + rec, 1e-12)


# ------------------------------------------------------------------ data
def load_days():
    days = []
    for d in DAYS:
        z = np.load(DATA / f"BTCUSDT_{d}.npz")
        days.append({k: z[k] for k in z.files} | {"day": d})
    return days


def prepare(days, train_ids):
    daily_vol = np.mean([(days[i]["buy_vol"] + days[i]["sell_vol"]).sum() for i in train_ids])
    bucket = daily_vol / L.VPIN_BUCKETS_PER_DAY
    for d in days:
        fut_fn, past_fn = L.LABELS[LABEL]
        d["tox"] = fut_fn(d["mid"], d["buy_vol"], d["sell_vol"])
        d["past_tox"] = past_fn(d["mid"], d["buy_vol"], d["sell_vol"])
        d["vpin"] = L.vpin_series(d["buy_vol"], d["sell_vol"], bucket)
        d["H"] = L.handcrafted_features(d["book"], d["mid"], d["buy_vol"], d["sell_vol"], d["vpin"], d["past_tox"])
        if "X_lob" not in d:   # causal rolling z-score removes day-level regime differences in the book
            d["X_lob"] = L.rolling_zscore(L.lob_features(d["book"], d["mid"], d["buy_vol"], d["sell_vol"], with_trades=True))
        # flow-context channels (absolute level, globally standardised later): 60-s log volume,
        # 60-s volatility, |60-s return|, spread, VPIN
        ctx = np.nan_to_num(d["H"][:, [5, 8, 7, 0, 9]], nan=0.0).astype(np.float32)
        d["X_full"] = np.concatenate([d["X_lob"], ctx], axis=1)
        n = len(d["mid"])
        t = np.arange(n)
        ok = (t >= WINDOW - 1) & np.isfinite(d["tox"]) & np.isfinite(d["vpin"]) & np.isfinite(d["past_tox"])
        d["valid"] = np.where(ok)[0]
    thr = np.quantile(np.concatenate([days[i]["tox"][days[i]["valid"]] for i in train_ids]), TOXIC_QUANTILE)
    for d in days:
        d["y"] = (d["tox"] >= thr).astype(np.int64)
    return thr, bucket


class WindowSet(torch.utils.data.Dataset):
    def __init__(self, days, ids, stride, feat_idx, mean, std):
        self.items = []
        self.X = {}
        for i in ids:
            d = days[i]
            self.X[i] = torch.from_numpy(np.clip((d["X_full"][:, feat_idx] - mean) / std, -10, 10).astype(np.float32))
            for t in d["valid"][::stride]:
                self.items.append((i, int(t), int(d["y"][t])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, k):
        i, t, y = self.items[k]
        return self.X[i][t - WINDOW + 1:t + 1], y


def predict(model, loader):
    model.eval(); out = []
    with torch.no_grad():
        for xb, _ in loader:
            logits, _ = model(xb.to(DEVICE))
            out.append(torch.softmax(logits[:, 0, :], dim=-1)[:, 1].cpu().numpy())
    return np.concatenate(out)


def train_deep(days, train_ids, val_id, test_id, feat_idx, name, fold):
    set_seed()
    Xtr = np.concatenate([days[i]["X_full"][days[i]["valid"]][:, feat_idx] for i in train_ids])
    mean = Xtr.mean(0); std = Xtr.std(0) + 1e-6
    tr = WindowSet(days, train_ids, TRAIN_STRIDE, feat_idx, mean, std)
    va = WindowSet(days, [val_id], VAL_STRIDE, feat_idx, mean, std)
    te = WindowSet(days, [test_id], 1, feat_idx, mean, std)
    g = torch.Generator(); g.manual_seed(SEED)
    tl = torch.utils.data.DataLoader(tr, batch_size=BATCH, shuffle=True, generator=g)
    vl = torch.utils.data.DataLoader(va, batch_size=1024)
    el = torch.utils.data.DataLoader(te, batch_size=1024)

    ytr = np.array([it[2] for it in tr.items])
    w = torch.tensor([len(ytr) / (2 * (ytr == 0).sum()), len(ytr) / (2 * (ytr == 1).sum())], dtype=torch.float32)
    model = TemporalCNNGRUAttention(input_features=len(mean), num_horizons=1, num_classes=2, dropout=DROPOUT).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    crit = nn.CrossEntropyLoss(weight=w.to(DEVICE))
    yva = np.array([it[2] for it in va.items])

    ckpt = OUT / f"fold{fold}_{name}.pt"
    best, bad, log = -1.0, 0, []
    for ep in range(EPOCHS):
        model.train(); t0 = time.time(); tot = 0.0
        for xb, yb in tl:
            opt.zero_grad()
            logits, _ = model(xb.to(DEVICE))
            loss = crit(logits[:, 0, :], yb.to(DEVICE))
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
            tot += loss.item() * len(yb)
        auc = roc_auc(yva, predict(model, vl))
        log.append({"epoch": ep + 1, "train_loss": tot / len(tr), "val_auc": auc})
        print(f"  fold {fold} {name} epoch {ep + 1}: loss {tot / len(tr):.4f} val AUC {auc:.4f} ({time.time() - t0:.0f}s)", flush=True)
        if auc > best:
            best, bad = auc, 0
            torch.save({"model_state_dict": {k: v.cpu() for k, v in model.state_dict().items()},
                        "mean": mean, "std": std, "feat_idx": list(map(int, feat_idx)),
                        "epoch": ep + 1, "val_auc": auc}, ckpt)
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    state = torch.load(ckpt, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    val_full = WindowSet(days, [val_id], 1, feat_idx, mean, std)
    s_val = predict(model, torch.utils.data.DataLoader(val_full, batch_size=1024))
    s_test = predict(model, el)
    return s_val, s_test, state["epoch"], log


def train_logistic(days, train_ids, val_id, test_id):
    Xtr = np.concatenate([days[i]["H"][days[i]["valid"]] for i in train_ids])
    ytr = np.concatenate([days[i]["y"][days[i]["valid"]] for i in train_ids])
    mean = Xtr.mean(0); std = Xtr.std(0) + 1e-9
    X = torch.tensor((Xtr - mean) / std, dtype=torch.float64); y = torch.tensor(ytr, dtype=torch.float64)
    pw = (ytr == 0).sum() / (ytr == 1).sum()
    lin = nn.Linear(X.shape[1], 1).double()
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=200)
    lossf = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pw, dtype=torch.float64))

    def closure():
        opt.zero_grad()
        loss = lossf(lin(X)[:, 0], y) + 1e-3 * (lin.weight ** 2).sum()
        loss.backward(); return loss
    opt.step(closure)

    def score(i):
        Xs = torch.tensor((days[i]["H"][days[i]["valid"]] - mean) / std, dtype=torch.float64)
        with torch.no_grad():
            return torch.sigmoid(lin(Xs)[:, 0]).numpy()
    coef = dict(zip(L.HANDCRAFTED_NAMES, lin.weight.detach().numpy()[0].round(4).tolist()))
    return score(val_id), score(test_id), coef


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("Device:", DEVICE, "| label:", LABEL, "| output:", OUT)
    days = load_days()
    rows, meta = [], {}
    full = FEATS["lob_trades"]; lob_only = FEATS["lob_only"]
    for fold, (train_ids, val_id, test_id) in FOLDS.items():
        thr, bucket = prepare(days, train_ids)
        dv, dt = days[val_id], days[test_id]
        yv, yt = dv["y"][dv["valid"]], dt["y"][dt["valid"]]
        meta[fold] = {"train": [DAYS[i] for i in train_ids], "val": DAYS[val_id], "test": DAYS[test_id],
                      "toxicity_threshold_bps": float(thr), "vpin_bucket_btc": float(bucket),
                      "test_windows": int(len(yt)), "test_toxic_rate": float(yt.mean()),
                      "val_toxic_rate": float(yv.mean())}
        print(f"\nFold {fold}: threshold {thr:.3f} bps, VPIN bucket {bucket:.1f} BTC, "
              f"test toxic rate {yt.mean():.3f} ({len(yt):,} windows)", flush=True)

        scores = {
            "VPIN": (dv["vpin"][dv["valid"]], dt["vpin"][dt["valid"]]),
            "Past toxicity (persistence)": (dv["past_tox"][dv["valid"]], dt["past_tox"][dt["valid"]]),
        }
        sv, st, coef = train_logistic(days, train_ids, val_id, test_id)
        scores["Logistic regression (11 features)"] = (sv, st)
        meta[fold]["logistic_coefficients"] = coef

        for name, sl, tag in [("CNN-BiGRU-Attention (LOB + trades)", full, "lob_trades"),
                              ("CNN-BiGRU-Attention (LOB only)", lob_only, "lob_only")]:
            sv, st, ep, log = train_deep(days, train_ids, val_id, test_id, sl, tag, fold)
            scores[name] = (sv, st)
            meta[fold][f"{tag}_best_epoch"] = ep
            meta[fold][f"{tag}_log"] = log
            np.savez_compressed(OUT / f"fold{fold}_{tag}_test_scores.npz", score=st, y=yt, t=dt["valid"])

        for name, (sv_, st_) in scores.items():
            thr_f1 = best_f1_threshold(yv, sv_)
            p, r, f1 = threshold_metrics(yt, st_, thr_f1)
            rows.append({"fold": fold, "model": name, "test_day": DAYS[test_id],
                         "auc_roc": round(roc_auc(yt, st_), 4), "pr_auc": round(average_precision(yt, st_), 4),
                         "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
                         "base_rate": round(float(yt.mean()), 4)})
            print(f"  {name:40s} AUC {rows[-1]['auc_roc']:.4f}  PR-AUC {rows[-1]['pr_auc']:.4f}  F1 {rows[-1]['f1']:.4f}", flush=True)
        np.savez_compressed(OUT / f"fold{fold}_baseline_test_scores.npz", vpin=dt["vpin"][dt["valid"]],
                            past_tox=dt["past_tox"][dt["valid"]], y=yt, t=dt["valid"],
                            tox=dt["tox"][dt["valid"]])

    with open(OUT / "toxicity_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    with open(OUT / "toxicity_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("\nSaved", OUT / "toxicity_results.csv")


if __name__ == "__main__":
    main()
