"""Ticket-level lottery EV against TRUE Polymarket resolutions, per BTC reference.

The original analyze_lottery_tickets.py scored tickets against outcomes
inferred from recorded ticks -- which the mixed-source strike bug corrupted
(42/320 wrong). This version scores every ticket against the actual
resolutions (fetch_true_outcomes.py) and runs on a source-consistent bench
copy (chainlink_ref or binance_ref), so the distance-to-strike each ticket
sees matches that reference's own strike.

Ticket sim (same as original): for each side, while price <= max_price and
seconds_left >= min_seconds_left, buy $1 at most once per spacing_s.
Payoff = 1/price - 1 if that side TRULY won, else -1.

Reports a distance-threshold sweep (keep tickets with |BTC-strike|% <= T at
purchase) plus the unfiltered baseline.

Usage:
    python scripts/analysis/analyze_lottery_tickets_true_outcomes.py \
        --markets-folder <folder> --true-outcomes-csv <csv> [--label name]
"""

import argparse
import csv as _csv
from pathlib import Path

import numpy as np
import pandas as pd

USECOLS = [
    "unix_time", "seconds_left", "up_price", "down_price",
    "btc_binance", "btc_chainlink", "price_to_beat",
]

THRESHOLDS = [0.01, 0.02, 0.03, 0.05, 0.08, 0.12, 0.20, None]


def market_tickets(csv_path: Path, true_outcome: str, max_price: float,
                   min_seconds_left: float, spacing_s: float) -> list[dict]:
    df = pd.read_csv(csv_path, usecols=USECOLS)
    btc = df["btc_chainlink"].where(df["btc_chainlink"].notna(), df["btc_binance"])
    df = df.assign(btc=btc)
    ptb = df["price_to_beat"].dropna()
    if df["btc"].dropna().empty or ptb.empty:
        return []
    strike = ptb.iloc[-1]

    times = df["unix_time"].to_numpy()
    btc_vals = df["btc"].to_numpy()
    seconds_left = df["seconds_left"].to_numpy()

    tickets = []
    for side in ("up", "down"):
        prices = df[f"{side}_price"].to_numpy()
        last_buy_t = -np.inf
        for i in range(len(df)):
            price = prices[i]
            if not (0 < price <= max_price):
                continue
            if seconds_left[i] < min_seconds_left:
                continue
            if times[i] - last_buy_t < spacing_s:
                continue
            if np.isnan(btc_vals[i]):
                continue
            last_buy_t = times[i]
            dist_pct = abs(btc_vals[i] - strike) / strike * 100.0
            won = side == true_outcome
            tickets.append({
                "market": csv_path.stem,
                "side": side,
                "price": price,
                "dist_pct": dist_pct,
                "seconds_left": seconds_left[i],
                "payoff": (1.0 / price - 1.0) if won else -1.0,
                "won": won,
            })
    return tickets


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--markets-folder", required=True)
    parser.add_argument("--true-outcomes-csv", required=True)
    parser.add_argument("--label", default="")
    parser.add_argument("--max-price", type=float, default=0.025)
    parser.add_argument("--min-seconds-left", type=float, default=5.0)
    parser.add_argument("--ticket-spacing-s", type=float, default=10.0)
    args = parser.parse_args()

    truth = {}
    with open(args.true_outcomes_csv, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            outcome = (row.get("true_outcome") or "").strip().lower()
            if outcome in ("up", "down"):
                truth[row["slug"]] = outcome

    folder = Path(args.markets_folder)
    tickets = []
    skipped = 0
    for path in sorted(folder.glob("*.csv")):
        outcome = truth.get(path.stem)
        if outcome is None:
            skipped += 1
            continue
        tickets.extend(market_tickets(
            path, outcome, args.max_price, args.min_seconds_left, args.ticket_spacing_s
        ))

    df = pd.DataFrame(tickets)
    label = args.label or folder.name
    print(f"=== {label}: {len(df)} tickets across {df['market'].nunique() if len(df) else 0} markets "
          f"(skipped {skipped} without truth) ===")
    if df.empty:
        return
    winners = df[df["won"]]
    print(f"winning tickets: {len(winners)} in {winners['market'].nunique()} markets; "
          f"win-market list: {sorted(winners['market'].unique())}")
    print(f"\ndistance-threshold sweep (keep tickets with dist_pct <= T):")
    print(f"{'T':>8} {'n':>6} {'wins':>5} {'win%':>6} {'total_EV':>9} {'EV/ticket':>10}")
    for t in THRESHOLDS:
        sub = df if t is None else df[df["dist_pct"] <= t]
        if sub.empty:
            print(f"{str(t):>8} {0:>6}")
            continue
        ev = sub["payoff"].sum()
        print(f"{('inf' if t is None else t):>8} {len(sub):>6} {int(sub['won'].sum()):>5} "
              f"{100 * sub['won'].mean():>5.1f}% {ev:>9.1f} {ev / len(sub):>10.3f}")


if __name__ == "__main__":
    main()
