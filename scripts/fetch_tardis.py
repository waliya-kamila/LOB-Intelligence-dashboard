"""Download the free Tardis.dev sample files used in the toxicity study.

Binance USD-M futures BTCUSDT, first day of six months in 2024:
25-level order-book snapshots and trades (with aggressor side).
Tardis.dev provides first-day-of-month data free of charge without an API key.
"""
from pathlib import Path
import subprocess

DAYS = ["2024-03-01", "2024-04-01", "2024-05-01", "2024-07-01", "2024-08-01", "2024-10-01"]
TYPES = ["book_snapshot_25", "trades"]
OUT = Path(__file__).resolve().parent.parent / "data_crypto" / "raw"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for day in DAYS:
        y, m, d = day.split("-")
        for t in TYPES:
            dest = OUT / f"BTCUSDT_{t}_{day}.csv.gz"
            if dest.exists() and dest.stat().st_size > 0:
                print("exists", dest.name)
                continue
            url = f"https://datasets.tardis.dev/v1/binance-futures/{t}/{y}/{m}/{d}/BTCUSDT.csv.gz"
            tmp = dest.with_suffix(".part")
            subprocess.run(["curl", "-sSL", "--fail", "--retry", "3", "-o", str(tmp), url], check=True)
            tmp.rename(dest)
            print("downloaded", dest.name, round(dest.stat().st_size / 1e6, 1), "MB")


if __name__ == "__main__":
    main()
