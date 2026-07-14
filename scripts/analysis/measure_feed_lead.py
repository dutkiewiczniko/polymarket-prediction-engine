"""Measure how much the recorded Binance feed leads the recorded Chainlink feed.

Two independent estimates per market, using the live-recorded market CSVs
(btc_binance + btc_chainlink sampled every poll tick):

1. Shift-scan correlation: grid both series to 0.5s, first-difference, and
   find the shift s (0..MAX_SHIFT_S) maximizing corr(chainlink_diff(t),
   binance_diff(t - s)). If Binance moves first, the best shift is > 0.
2. Per-update value-match lag: for each Chainlink update to a new value V,
   find the most recent earlier time Binance first traded through V (coming
   from the previous Chainlink value's side). Lag = update time - crossing
   time. Robust to the small Binance-vs-Chainlink basis because it uses
   crossings, not exact equality.

Usage:
    python scripts/analysis/measure_feed_lead.py --markets-folder simulator_ready_live_20260713
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

MAX_SHIFT_S = 30.0
GRID_S = 0.5


def best_shift_corr(times, binance, chainlink):
    t0, t1 = times[0], times[-1]
    if t1 - t0 < 60:
        return None
    grid = np.arange(t0, t1, GRID_S)
    b = np.interp(grid, times[~np.isnan(binance)], binance[~np.isnan(binance)])
    c = np.interp(grid, times[~np.isnan(chainlink)], chainlink[~np.isnan(chainlink)])
    db, dc = np.diff(b), np.diff(c)
    if db.std() == 0 or dc.std() == 0:
        return None
    best_s, best_r = None, -2.0
    for shift_ticks in range(0, int(MAX_SHIFT_S / GRID_S) + 1):
        if shift_ticks == 0:
            r = np.corrcoef(dc, db)[0, 1]
        else:
            r = np.corrcoef(dc[shift_ticks:], db[:-shift_ticks])[0, 1]
        if np.isfinite(r) and r > best_r:
            best_r, best_s = r, shift_ticks * GRID_S
    return best_s, best_r


def update_value_match_lags(times, binance, chainlink):
    """For each chainlink update old->new, when did binance first cross new
    (from old's side) since the previous update? Returns list of lags (s)."""
    lags = []
    cl_idx = np.where(~np.isnan(chainlink))[0]
    if len(cl_idx) < 3:
        return lags
    prev_i = cl_idx[0]
    prev_v = chainlink[prev_i]
    for i in cl_idx[1:]:
        v = chainlink[i]
        if v == prev_v:
            continue
        window = (times >= times[prev_i]) & (times <= times[i])
        wt, wb = times[window], binance[window]
        ok = ~np.isnan(wb)
        wt, wb = wt[ok], wb[ok]
        if len(wb) > 1:
            crossed = wb >= v if v > prev_v else wb <= v
            hits = np.where(crossed)[0]
            if len(hits):
                lags.append(times[i] - wt[hits[0]])
        prev_i, prev_v = i, v
    return lags


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--markets-folder", default="simulator_ready_live_20260713")
    parser.add_argument("--market-pattern", default="btc-updown-5m-*.csv")
    args = parser.parse_args()

    files = sorted(Path(args.markets_folder).glob(args.market_pattern))
    shifts, corrs, all_lags, n_updates = [], [], [], 0
    for f in files:
        df = pd.read_csv(f, usecols=["unix_time", "btc_binance", "btc_chainlink"])
        times = df["unix_time"].to_numpy(float)
        b = df["btc_binance"].to_numpy(float)
        c = df["btc_chainlink"].to_numpy(float)
        res = best_shift_corr(times, b, c)
        if res is not None:
            shifts.append(res[0]); corrs.append(res[1])
        lags = update_value_match_lags(times, b, c)
        n_updates += len(lags)
        all_lags.extend(lags)

    shifts, corrs, all_lags = np.array(shifts), np.array(corrs), np.array(all_lags)
    print(f"{len(files)} markets | shift-scan on {len(shifts)} | {n_updates} chainlink updates value-matched")
    if len(shifts):
        print("\n--- shift-scan (positive = binance leads) ---")
        print(f"median best shift {np.median(shifts):+.1f}s | mean {shifts.mean():+.1f}s | "
              f"P(shift>0) {(shifts > 0).mean():.2f} | median corr {np.median(corrs):.3f}")
        print("shift deciles:", np.round(np.percentile(shifts, [10, 25, 50, 75, 90]), 1))
    if len(all_lags):
        print("\n--- per-update value-match lag (positive = binance crossed first) ---")
        print(f"median {np.median(all_lags):+.2f}s | mean {all_lags.mean():+.2f}s | "
              f"P(lag>1s) {(all_lags > 1).mean():.2f}")
        print("lag deciles:", np.round(np.percentile(all_lags, [10, 25, 50, 75, 90]), 2))


if __name__ == "__main__":
    main()
