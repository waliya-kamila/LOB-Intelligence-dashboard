"""
Convert raw Tardis.dev files into compact 1-second arrays per trading day.

Input  (data_crypto/raw):
    BTCUSDT_book_snapshot_25_<day>.csv.gz   25-level order-book snapshots
    BTCUSDT_trades_<day>.csv.gz             trades with aggressor side

Output (data_crypto/processed/BTCUSDT_<day>.npz):
    book      (86400, 40)  last snapshot in each second, top 10 levels:
                           [ask_p1..ask_p10, ask_q1..ask_q10, bid_p1..bid_p10, bid_q1..bid_q10]
    mid       (86400,)     mid-price at the end of each second
    buy_vol   (86400,)     aggressive buy volume (BTC) in each second
    sell_vol  (86400,)     aggressive sell volume (BTC) in each second
    n_trades  (86400,)     number of trades in each second

Usage:
    python -m src.toxicity.prepare
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
RAW = ROOT / "data_crypto" / "raw"
OUT = ROOT / "data_crypto" / "processed"

DAYS = ["2024-03-01", "2024-04-01", "2024-05-01", "2024-07-01", "2024-08-01", "2024-10-01"]
LEVELS = 10
SECONDS = 86_400

BOOK_COLS = (
    [f"asks[{i}].price" for i in range(LEVELS)]
    + [f"asks[{i}].amount" for i in range(LEVELS)]
    + [f"bids[{i}].price" for i in range(LEVELS)]
    + [f"bids[{i}].amount" for i in range(LEVELS)]
)


def day_start_us(day):
    return int(pd.Timestamp(day, tz="UTC").value // 1_000)


def load_book(day):
    path = RAW / f"BTCUSDT_book_snapshot_25_{day}.csv.gz"
    t0 = day_start_us(day)
    parts = []
    for chunk in pd.read_csv(path, usecols=["timestamp"] + BOOK_COLS, chunksize=500_000):
        sec = ((chunk["timestamp"].to_numpy() - t0) // 1_000_000).astype(np.int64)
        chunk = chunk[BOOK_COLS]
        chunk.index = sec
        # last snapshot within each second
        parts.append(chunk[~chunk.index.duplicated(keep="last")])
    book = pd.concat(parts)
    book = book[~book.index.duplicated(keep="last")]
    book = book[(book.index >= 0) & (book.index < SECONDS)]
    book = book.reindex(np.arange(SECONDS)).ffill().bfill()
    return book.to_numpy(dtype=np.float64)


def load_trades(day):
    path = RAW / f"BTCUSDT_trades_{day}.csv.gz"
    t0 = day_start_us(day)
    buy = np.zeros(SECONDS)
    sell = np.zeros(SECONDS)
    count = np.zeros(SECONDS)
    for chunk in pd.read_csv(path, usecols=["timestamp", "side", "amount"], chunksize=1_000_000):
        sec = ((chunk["timestamp"].to_numpy() - t0) // 1_000_000).astype(np.int64)
        ok = (sec >= 0) & (sec < SECONDS)
        sec = sec[ok]
        amt = chunk["amount"].to_numpy()[ok]
        is_buy = (chunk["side"].to_numpy()[ok] == "buy")
        np.add.at(buy, sec[is_buy], amt[is_buy])
        np.add.at(sell, sec[~is_buy], amt[~is_buy])
        np.add.at(count, sec, 1)
    return buy, sell, count


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for day in DAYS:
        out = OUT / f"BTCUSDT_{day}.npz"
        if out.exists():
            print("exists", out.name)
            continue
        if not all((RAW / f"BTCUSDT_{t}_{day}.csv.gz").exists() for t in ("book_snapshot_25", "trades")):
            print("raw files not yet available:", day)
            continue
        book = load_book(day)
        mid = (book[:, 0] + book[:, 2 * LEVELS]) / 2.0
        buy, sell, count = load_trades(day)
        assert np.isfinite(book).all() and (book[:, 0] > book[:, 2 * LEVELS]).all(), "crossed or missing book"
        np.savez_compressed(out, book=book.astype(np.float64), mid=mid, buy_vol=buy,
                            sell_vol=sell, n_trades=count)
        print(f"{day}: mid {mid.min():.1f}-{mid.max():.1f}  trades {int(count.sum()):,}  "
              f"volume {buy.sum() + sell.sum():,.0f} BTC  buy share {buy.sum() / (buy.sum() + sell.sum()):.3f}")


if __name__ == "__main__":
    main()
