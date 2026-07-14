"""Compare Binance vs Chainlink as the BTC reference across bench markets.

For each market CSV, computes:
  * coverage: how many ticks carry each feed, and the largest silent gap
  * start price per source (first valid sample) and its delta vs recorded price_to_beat
  * implied outcome per source: last price vs that SAME source's start price
    (self-consistent referencing, per the "reference everything to one source" rule)
  * whether the two sources disagree on the outcome -- the real-money risk when
    deciding on Binance while Polymarket resolves on Chainlink.

Usage:
    python scripts/analysis/analyze_binance_vs_chainlink.py <market_folder>
"""

import csv
import statistics
import sys
from pathlib import Path


def parse_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def analyze_market(path: Path):
    binance = []  # (unix_time, price)
    chainlink = []
    ptb = None
    n_ticks = 0
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            n_ticks += 1
            t = parse_float(row.get("unix_time"))
            b = parse_float(row.get("btc_binance"))
            c = parse_float(row.get("btc_chainlink"))
            if t is None:
                continue
            if b is not None:
                binance.append((t, b))
            if c is not None:
                chainlink.append((t, c))
            if ptb is None:
                ptb = parse_float(row.get("price_to_beat"))

    def max_gap(series):
        if len(series) < 2:
            return None
        return max(b[0] - a[0] for a, b in zip(series, series[1:]))

    def implied_outcome(series):
        if len(series) < 2:
            return None
        return "up" if series[-1][1] >= series[0][1] else "down"

    # Recorded-reference outcome: chainlink-first last price vs recorded ptb
    # (mirrors simulator.replay.infer_final_outcome).
    last_ref = chainlink[-1][1] if chainlink else (binance[-1][1] if binance else None)
    recorded_outcome = None
    if last_ref is not None and ptb is not None:
        recorded_outcome = "up" if last_ref >= ptb else "down"

    return {
        "slug": path.stem,
        "n_ticks": n_ticks,
        "binance_n": len(binance),
        "chainlink_n": len(chainlink),
        "binance_max_gap": max_gap(binance),
        "chainlink_max_gap": max_gap(chainlink),
        "binance_start": binance[0][1] if binance else None,
        "chainlink_start": chainlink[0][1] if chainlink else None,
        "ptb": ptb,
        "binance_outcome": implied_outcome(binance),
        "chainlink_outcome": implied_outcome(chainlink),
        "recorded_outcome": recorded_outcome,
        "start_delta": (binance[0][1] - chainlink[0][1]) if binance and chainlink else None,
    }


def main():
    folder = Path(sys.argv[1] if len(sys.argv) > 1 else "simulator_ready_markets_liquidity_all_20260519")
    results = [analyze_market(p) for p in sorted(folder.glob("*.csv"))]
    n = len(results)
    print(f"markets analyzed: {n}\n")

    no_binance = [r for r in results if r["binance_n"] == 0]
    no_chainlink = [r for r in results if r["chainlink_n"] == 0]
    print(f"markets with NO binance samples:   {len(no_binance)}")
    print(f"markets with NO chainlink samples: {len(no_chainlink)}")
    for r in no_chainlink[:10]:
        print(f"  no-chainlink: {r['slug']}")

    both = [r for r in results if r["binance_n"] and r["chainlink_n"]]

    b_gaps = [r["binance_max_gap"] for r in both if r["binance_max_gap"] is not None]
    c_gaps = [r["chainlink_max_gap"] for r in both if r["chainlink_max_gap"] is not None]
    print(f"\nlargest silent gap within a market (seconds):")
    print(f"  binance   median={statistics.median(b_gaps):.1f}  p95={sorted(b_gaps)[int(len(b_gaps)*0.95)]:.1f}  max={max(b_gaps):.1f}")
    print(f"  chainlink median={statistics.median(c_gaps):.1f}  p95={sorted(c_gaps)[int(len(c_gaps)*0.95)]:.1f}  max={max(c_gaps):.1f}")
    print(f"  markets where binance gap > 30s:   {sum(1 for g in b_gaps if g > 30)}")
    print(f"  markets where chainlink gap > 30s: {sum(1 for g in c_gaps if g > 30)}")

    deltas = [r["start_delta"] for r in both if r["start_delta"] is not None]
    abs_deltas = [abs(d) for d in deltas]
    print(f"\nstart price delta (binance - chainlink), USD:")
    print(f"  median={statistics.median(deltas):+.2f}  mean={statistics.mean(deltas):+.2f}")
    print(f"  |delta| median={statistics.median(abs_deltas):.2f}  p95={sorted(abs_deltas)[int(len(abs_deltas)*0.95)]:.2f}  max={max(abs_deltas):.2f}")

    # The headline number: self-consistent outcome disagreement.
    comparable = [r for r in both if r["binance_outcome"] and r["chainlink_outcome"]]
    disagree = [r for r in comparable if r["binance_outcome"] != r["chainlink_outcome"]]
    print(f"\nself-consistent outcome (last vs own start):")
    print(f"  comparable markets: {len(comparable)}")
    print(f"  binance vs chainlink DISAGREE: {len(disagree)} ({100*len(disagree)/len(comparable):.1f}%)")
    for r in disagree:
        print(f"    {r['slug']}: binance={r['binance_outcome']} chainlink={r['chainlink_outcome']} "
              f"b_start={r['binance_start']:.2f} c_start={r['chainlink_start']:.2f} ptb={r['ptb']}")

    # Binance-world outcome vs the outcome the backtests actually paid on.
    comparable2 = [r for r in comparable if r["recorded_outcome"]]
    disagree2 = [r for r in comparable2 if r["binance_outcome"] != r["recorded_outcome"]]
    print(f"\nbinance-world outcome vs recorded (ptb-based, what backtests pay on):")
    print(f"  DISAGREE: {len(disagree2)} of {len(comparable2)} ({100*len(disagree2)/len(comparable2):.1f}%)")
    for r in disagree2:
        print(f"    {r['slug']}: binance={r['binance_outcome']} recorded={r['recorded_outcome']}")


if __name__ == "__main__":
    main()
