"""
Toxicity labels, VPIN and model features on the 1-second grid.

Toxicity (adverse selection of liquidity providers)
---------------------------------------------------
Every trade is initiated by an aggressor (sign s = +1 buy, -1 sell) against a
resting limit order supplied by a liquidity provider. The liquidity provider's
loss per unit of volume is the trade's *markout*:

    m = s * (mid[t + TAU] - mid[t-]) / mid[t-]          (in basis points)

A positive markout means the price moved in the aggressor's favour, i.e. the
liquidity provider was adversely selected. Aggregating over all trades executed
in the next H seconds after decision time t gives the volume-weighted toxicity

    TOX(t) = sum_i v_i * m_i / sum_i v_i,   trades i in (t, t + H].

Flow is labelled toxic when TOX(t) exceeds the 80th percentile of the training
distribution. The prediction at time t uses only information up to t, so the
model must detect toxic flow *before* it is traded (early detection).

VPIN (Easley, Lopez de Prado & O'Hara, 2012)
--------------------------------------------
Trades are grouped into equal-volume buckets of size V; VPIN is the mean of
|V_buy - V_sell| / V over the last N buckets. The exact aggressor side is
known here, so no bulk-volume classification is needed. VPIN(t) is the value
after the last bucket completed by time t.
"""

import numpy as np

H = 30            # seconds of future flow evaluated
TAU = 30          # markout horizon in seconds
LEVELS = 10
VPIN_BUCKETS_PER_DAY = 1000   # ~86 s average bucket; 50-bucket window warms up in ~70 min
VPIN_WINDOW = 50


def markout_per_second(mid, buy_vol, sell_vol, tau=TAU):
    """Sum of v_i * m_i for the trades executed in each second (bps * BTC)."""
    n = len(mid)
    pre = np.empty(n); pre[0] = mid[0]; pre[1:] = mid[:-1]        # mid just before the second
    post_idx = np.minimum(np.arange(n) + tau, n - 1)
    ret_bps = (mid[post_idx] - pre) / pre * 1e4
    return ret_bps * (buy_vol - sell_vol)


def future_toxicity(mid, buy_vol, sell_vol, h=H, tau=TAU):
    """TOX(t) over trades in (t, t+h]; NaN where the label is incomplete."""
    sv = markout_per_second(mid, buy_vol, sell_vol, tau)
    vol = buy_vol + sell_vol
    csv = np.concatenate([[0.0], np.cumsum(sv)])
    cv = np.concatenate([[0.0], np.cumsum(vol)])
    n = len(mid)
    t = np.arange(n)
    hi = np.minimum(t + 1 + h, n)
    num = csv[hi] - csv[t + 1]
    den = cv[hi] - cv[t + 1]
    tox = np.where(den > 0, num / np.maximum(den, 1e-12), 0.0)
    tox[n - h - tau:] = np.nan            # future window / markout not fully observed
    return tox


def past_toxicity(mid, buy_vol, sell_vol, h=H, tau=TAU):
    """Realised toxicity of trades in (t-h-tau, t-tau]: fully observable at time t."""
    fut = future_toxicity(mid, buy_vol, sell_vol, h, tau)
    out = np.full_like(fut, np.nan)
    shift = h + tau
    # fut[t - shift] covers trades in (t-shift, t-tau] with markouts ending by t
    out[shift:] = fut[:-shift]
    return out


def vpin_series(buy_vol, sell_vol, bucket_size, window=VPIN_WINDOW):
    """VPIN value available at the end of each second (NaN until `window` buckets exist)."""
    n = len(buy_vol)
    out = np.full(n, np.nan)
    imbalances = []
    cur_b = cur_s = 0.0
    current = np.nan
    for t in range(n):
        b, s = buy_vol[t], sell_vol[t]
        total = b + s
        while total > 0:
            room = bucket_size - (cur_b + cur_s)
            take = min(room, total)
            frac = take / total
            cur_b += b * frac; cur_s += s * frac
            b -= b * frac; s -= s * frac; total -= take
            if cur_b + cur_s >= bucket_size - 1e-12:
                imbalances.append(abs(cur_b - cur_s) / bucket_size)
                cur_b = cur_s = 0.0
                if len(imbalances) >= window:
                    current = float(np.mean(imbalances[-window:]))
        out[t] = current
    return out


def lob_features(book, mid, buy_vol, sell_vol, with_trades=True):
    """Per-second model inputs.

    Book (40): price distance of each level from mid in bps (ask/bid) and
    log volume at each level. Plus 1-second mid return (bps).
    Trades (3, optional): log aggressive buy volume, log aggressive sell
    volume and signed trade imbalance in the second.
    """
    ask_p, ask_q = book[:, 0:LEVELS], book[:, LEVELS:2 * LEVELS]
    bid_p, bid_q = book[:, 2 * LEVELS:3 * LEVELS], book[:, 3 * LEVELS:4 * LEVELS]
    m = mid[:, None]
    feats = [
        (ask_p - m) / m * 1e4,
        np.log1p(ask_q),
        (m - bid_p) / m * 1e4,
        np.log1p(bid_q),
    ]
    ret = np.zeros(len(mid)); ret[1:] = (mid[1:] - mid[:-1]) / mid[:-1] * 1e4
    feats.append(ret[:, None])
    if with_trades:
        tot = buy_vol + sell_vol
        feats += [np.log1p(buy_vol)[:, None], np.log1p(sell_vol)[:, None],
                  np.where(tot > 0, (buy_vol - sell_vol) / np.maximum(tot, 1e-12), 0.0)[:, None]]
    return np.concatenate(feats, axis=1).astype(np.float32)


FEATURE_NAMES = (
    [f"ask_dist_L{i+1}" for i in range(LEVELS)] + [f"ask_vol_L{i+1}" for i in range(LEVELS)]
    + [f"bid_dist_L{i+1}" for i in range(LEVELS)] + [f"bid_vol_L{i+1}" for i in range(LEVELS)]
    + ["mid_return", "trade_buy_vol", "trade_sell_vol", "trade_imbalance"]
    + ["ctx_log_volume_60s", "ctx_volatility_60s", "ctx_abs_return_60s", "ctx_spread", "ctx_VPIN"]
)


def handcrafted_features(book, mid, buy_vol, sell_vol, vpin, past_tox, lookback=60):
    """Features for the logistic-regression baseline (observable at time t)."""
    ask_q = book[:, LEVELS:2 * LEVELS]; bid_q = book[:, 3 * LEVELS:4 * LEVELS]
    spread = (book[:, 0] - book[:, 2 * LEVELS]) / mid * 1e4
    l1_imb = (bid_q[:, 0] - ask_q[:, 0]) / (bid_q[:, 0] + ask_q[:, 0])
    depth_imb = (bid_q.sum(1) - ask_q.sum(1)) / (bid_q.sum(1) + ask_q.sum(1))
    cb = np.concatenate([[0], np.cumsum(buy_vol)]); cs = np.concatenate([[0], np.cumsum(sell_vol)])
    t = np.arange(len(mid)); lo = np.maximum(t + 1 - lookback, 0)
    b60 = cb[t + 1] - cb[lo]; s60 = cs[t + 1] - cs[lo]
    flow_imb = (b60 - s60) / np.maximum(b60 + s60, 1e-12)
    logvol = np.log1p(b60 + s60)
    lm = np.log(mid); r = np.zeros(len(mid)); r[1:] = np.diff(lm) * 1e4
    cr = np.concatenate([[0], np.cumsum(r)]); cr2 = np.concatenate([[0], np.cumsum(r ** 2)])
    ret60 = cr[t + 1] - cr[lo]
    vol60 = np.sqrt(np.maximum((cr2[t + 1] - cr2[lo]) / lookback, 0))
    X = np.column_stack([spread, l1_imb, depth_imb, flow_imb, np.abs(flow_imb), logvol,
                         ret60, np.abs(ret60), vol60, vpin, past_tox])
    return X.astype(np.float64)


HANDCRAFTED_NAMES = ["spread", "L1_imbalance", "depth_imbalance", "flow_imbalance_60s",
                     "abs_flow_imbalance_60s", "log_volume_60s", "return_60s", "abs_return_60s",
                     "volatility_60s", "VPIN", "past_toxicity"]


# ---------------------------------------------------------------------------
# Alternative label: aggregate adverse-selection loss of liquidity providers
# ---------------------------------------------------------------------------
# LPLOSS(t) = sum_i v_i * m_i over trades in (t, t+H]   (bps x BTC)
# TOX(t) above measures toxicity *intensity* (loss per unit traded);
# LPLOSS(t) measures toxicity *exposure* (total loss liquidity providers suffer),
# which is what a market maker must manage in practice.

def future_lp_loss(mid, buy_vol, sell_vol, h=H, tau=TAU):
    sv = markout_per_second(mid, buy_vol, sell_vol, tau)
    c = np.concatenate([[0.0], np.cumsum(sv)])
    n = len(mid)
    t = np.arange(n)
    out = c[np.minimum(t + 1 + h, n)] - c[t + 1]
    out[n - h - tau:] = np.nan
    return out


def past_lp_loss(mid, buy_vol, sell_vol, h=H, tau=TAU):
    fut = future_lp_loss(mid, buy_vol, sell_vol, h, tau)
    out = np.full_like(fut, np.nan)
    shift = h + tau
    out[shift:] = fut[:-shift]
    return out


LABELS = {
    "intensity": (future_toxicity, past_toxicity),
    "exposure": (future_lp_loss, past_lp_loss),
}


def rolling_zscore(X, window=3600, min_periods=300):
    """Causal z-score: each value relative to the mean/std of the previous `window` seconds.

    Uses only past data, so it is valid at prediction time. It removes slow
    day-to-day regime differences (price level, typical depth, typical volume)
    so the model learns *relative* order-book states that transfer across days.
    """
    X = X.astype(np.float64)
    n, f = X.shape
    c1 = np.vstack([np.zeros((1, f)), np.cumsum(X, axis=0)])
    c2 = np.vstack([np.zeros((1, f)), np.cumsum(X ** 2, axis=0)])
    t = np.arange(n)
    lo = np.maximum(t - window, 0)            # past window [t-window, t)
    cnt = (t - lo)[:, None].astype(np.float64)
    s1 = c1[t] - c1[lo]; s2 = c2[t] - c2[lo]
    mean = s1 / np.maximum(cnt, 1)
    var = np.maximum(s2 / np.maximum(cnt, 1) - mean ** 2, 0)
    Z = (X - mean) / (np.sqrt(var) + 1e-6)
    Z[t < min_periods] = 0.0
    return np.clip(Z, -10, 10).astype(np.float32)
