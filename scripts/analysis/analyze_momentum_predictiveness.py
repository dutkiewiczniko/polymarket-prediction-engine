"""Measure how often BTC momentum predicts the market's final outcome.

Momentum is a per-tick metric, so a "prediction" has to be pinned to a moment:
this script samples each window's momentum sign at fixed checkpoints of elapsed
market time (60s / 150s / 240s of a 300s market), computed exactly as the engine
would at the last valid-BTC tick at or before the checkpoint. It also reports a
majority-sign-over-the-market variant.

Momentum definitions match simulator/strategies.py:
- momentum_<W>s  = pct change of BTC now vs the latest sample at or before
  now - W seconds (momentum_pct)
- momentum_<N>t  = pct change vs exactly N recorded BTC ticks ago
  (momentum_pct_by_ticks); ticks with no BTC value are never recorded
BTC = btc_chainlink falling back to btc_binance, as in the engine.

Reported per window:
- pooled per-tick momentum distribution (for threshold calibration)
- sign alignment (favored side wins) at each checkpoint, overall and
  conditional on the champion-style threshold (0.0001% ticks, 0.03% 60/120s)
- majority-sign alignment
- favored win rate by |momentum| magnitude quartile at the mid checkpoint

Usage:
    python scripts/analysis/analyze_momentum_predictiveness.py \
        --markets-folder simulator_ready_markets_liquidity_all_20260519
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.strategies import momentum_pct, momentum_pct_by_ticks  # noqa: E402

TICK_WINDOWS = (1, 2)
TIME_WINDOWS_S = (60.0, 120.0)
CHECKPOINTS_S = (60.0, 150.0, 240.0)
# Champion strategy fires around these magnitudes (in percent, like pct_change).
CHAMPION_THRESHOLD = {"momentum_1t": 0.0001, "momentum_2t": 0.0001, "momentum_60s": 0.03, "momentum_120s": 0.03}
WINDOW_NAMES = [f"momentum_{n}t" for n in TICK_WINDOWS] + [f"momentum_{int(w)}s" for w in TIME_WINDOWS_S]


def load_market(csv_path: Path) -> dict | None:
    df = pd.read_csv(
        csv_path,
        usecols=["unix_time", "elapsed", "btc_binance", "btc_chainlink", "price_to_beat"],
    )
    btc = df["btc_chainlink"].where(df["btc_chainlink"].notna(), df["btc_binance"])
    ptb = df["price_to_beat"].dropna()
    valid = btc.notna()
    if not valid.any() or ptb.empty:
        return None
    return {
        "times": df.loc[valid, "unix_time"].to_numpy(dtype=float),
        "prices": btc[valid].to_numpy(dtype=float),
        "elapsed": df.loc[valid, "elapsed"].to_numpy(dtype=float),
        # infer_final_outcome: last BTC (chainlink fallback binance) vs last price_to_beat
        "outcome": "up" if btc[valid].iloc[-1] >= ptb.iloc[-1] else "down",
    }


def momentum_series(times: np.ndarray, prices: np.ndarray) -> dict[str, np.ndarray]:
    """Per-tick momentum for every window, NaN where the engine would return None."""
    out = {}
    for n in TICK_WINDOWS:
        mom = np.full(len(prices), np.nan)
        if len(prices) > n:
            mom[n:] = (prices[n:] - prices[:-n]) / prices[:-n] * 100.0
        out[f"momentum_{n}t"] = mom
    for window_s in TIME_WINDOWS_S:
        # latest sample at or before t - W == last index j with times[j] <= t - W
        prev_idx = np.searchsorted(times, times - window_s, side="right") - 1
        mom = np.full(len(prices), np.nan)
        has_prev = prev_idx >= 0
        prev = prices[np.clip(prev_idx, 0, None)]
        with np.errstate(invalid="ignore", divide="ignore"):
            mom[has_prev] = ((prices - prev) / prev * 100.0)[has_prev]
        out[f"momentum_{int(window_s)}s"] = mom
    return out


def verify_against_engine(times: np.ndarray, prices: np.ndarray, moms: dict[str, np.ndarray]) -> None:
    """Spot-check the vectorized momentum against the engine's reference functions."""
    rng = np.random.default_rng(0)
    for i in rng.choice(len(prices), size=min(25, len(prices)), replace=False):
        series = list(zip(times[: i + 1], prices[: i + 1]))
        for n in TICK_WINDOWS:
            expected = momentum_pct_by_ticks(series, n)
            got = moms[f"momentum_{n}t"][i]
            assert (expected is None and np.isnan(got)) or abs(expected - got) < 1e-12, f"{n}t mismatch at {i}"
        for window_s in TIME_WINDOWS_S:
            expected = momentum_pct(series, times[i], window_s)
            got = moms[f"momentum_{int(window_s)}s"][i]
            assert (expected is None and np.isnan(got)) or abs(expected - got) < 1e-12, f"{window_s}s mismatch at {i}"


def analyze_market(market: dict) -> tuple[dict, dict[str, np.ndarray]]:
    moms = momentum_series(market["times"], market["prices"])
    row = {"outcome": market["outcome"], "n_ticks": len(market["prices"])}
    for name, mom in moms.items():
        finite = mom[np.isfinite(mom)]
        pos, neg = (finite > 0).sum(), (finite < 0).sum()
        row[f"{name}_mean_abs"] = np.abs(finite).mean() if len(finite) else np.nan
        row[f"{name}_share_pos"] = pos / len(finite) if len(finite) else np.nan
        row[f"{name}_majority"] = "up" if pos > neg else ("down" if neg > pos else None)
        for cp in CHECKPOINTS_S:
            idx = np.searchsorted(market["elapsed"], cp, side="right") - 1
            row[f"{name}_at{int(cp)}s"] = mom[idx] if idx >= 0 else np.nan
    return row, moms


def sign_alignment(df: pd.DataFrame, col: str, min_abs: float = 0.0) -> tuple[float, int]:
    sub = df[df[col].notna() & (df[col].abs() > min_abs if min_abs else df[col] != 0)]
    if sub.empty:
        return np.nan, 0
    favored = np.where(sub[col] > 0, "up", "down")
    return (favored == sub["outcome"]).mean(), len(sub)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--markets-folder", default="simulator_ready_markets_liquidity_all_20260519")
    args = parser.parse_args()

    folder = Path(args.markets_folder)
    rows, pooled = [], {name: [] for name in WINDOW_NAMES}
    verified = False
    for csv_path in sorted(folder.glob("btc-updown-5m-*.csv")):
        market = load_market(csv_path)
        if market is None:
            continue
        row, moms = analyze_market(market)
        if not verified:
            verify_against_engine(market["times"], market["prices"], moms)
            verified = True
        rows.append({"market_file": csv_path.name, **row})
        for name, mom in moms.items():
            pooled[name].append(mom[np.isfinite(mom)])
    df = pd.DataFrame(rows)

    up_rate = (df["outcome"] == "up").mean()
    print(f"{len(df)} markets | base rate: up {up_rate:.1%} / down {1 - up_rate:.1%}")
    print(f"prediction sampled at elapsed checkpoints {[int(c) for c in CHECKPOINTS_S]}s of ~300s markets\n")

    for name in WINDOW_NAMES:
        ticks = np.concatenate(pooled[name])
        threshold = CHAMPION_THRESHOLD[name]
        print(f"=== {name} ===")
        print(
            "  pooled per-tick distribution (%): "
            + " ".join(f"p{int(q * 100)}={np.quantile(ticks, q):+.5f}" for q in (0.05, 0.25, 0.5, 0.75, 0.95))
            + f"  |mom|>{threshold:g}: {(np.abs(ticks) > threshold).mean():.1%} of ticks"
        )
        print("  sign alignment (favored side wins):")
        for cp in CHECKPOINTS_S:
            col = f"{name}_at{int(cp)}s"
            rate, n = sign_alignment(df, col)
            rate_thr, n_thr = sign_alignment(df, col, min_abs=threshold)
            print(f"    at {int(cp):>3}s: {rate:6.1%} (n={n:3d})   |mom|>{threshold:g}: {rate_thr:6.1%} (n={n_thr:3d})")
        majority = df[df[f"{name}_majority"].notna()]
        maj_rate = (majority[f"{name}_majority"] == majority["outcome"]).mean()
        print(f"    majority sign over market: {maj_rate:.1%} (n={len(majority)})")

        mid = f"{name}_at{int(CHECKPOINTS_S[1])}s"
        sub = df[df[mid].notna() & (df[mid] != 0)].copy()
        sub["abs_mag"] = sub[mid].abs()
        sub["mag_bucket"] = pd.qcut(sub["abs_mag"], 4, labels=["q1_weak", "q2", "q3", "q4_strong"], duplicates="drop")
        bucket = sub.groupby("mag_bucket", observed=True).apply(
            lambda g: pd.Series({
                "n": len(g),
                "mag_min": g["abs_mag"].min(),
                "mag_max": g["abs_mag"].max(),
                "favored_win_rate": (np.where(g[mid] > 0, "up", "down") == g["outcome"]).mean(),
            }),
            include_groups=False,
        )
        print(f"  |momentum| quartiles at {int(CHECKPOINTS_S[1])}s:")
        print(bucket.round(5).to_string())
        print()

    out_path = folder.parent / f"{folder.name}_momentum_outcomes.csv"
    df.to_csv(out_path, index=False)
    print(f"Per-market table: {out_path}")


if __name__ == "__main__":
    main()
