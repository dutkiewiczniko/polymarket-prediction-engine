"""What predicts a cheap side flipping, at the moment it is cheap?

For every first-touch of a cheap price level (side price <= L, default grid
0.02/0.05/0.10/0.15), compute BTC-state features AT THAT TICK and test how well
each separates flips (side truly won per Polymarket resolution) from non-flips:

  * deficit_pct    signed % BTC must move to put this side in the money
                   (up side: (strike-btc)/strike*100; down side mirrored;
                   negative = side already in the money despite cheap price)
  * mom_toward_Ws  BTC % change over the last W seconds, signed so that
                   POSITIVE = moving in the direction this side needs
  * vol_Ws_pct     BTC high-low range % over the last W seconds
  * seconds_left   time remaining at touch

Outcome scored ONLY from the true-outcomes CSV (real Polymarket resolutions),
never inferred from ticks (42/320 inference errors -- see memory notes).
Strike/BTC from the corrected official_ref bench.

Payoff model: hold to resolution ($1 buy: 1/p - 1 if flip else -1) -- the
threshold sweep in analyze_comeback_by_price_level_true_outcomes.py showed
hold-to-resolution is the right exit for the cheap zone.

Reported per level: feature quartile bins (n / flips / flip% / EV/$1), rank
AUC per feature (0.5 = useless, >0.5 = higher value -> more flips), the
deficit x momentum 2x2 interaction, and the flip list with features.

Usage:
    python scripts/analysis/analyze_flip_predictors_true_outcomes.py \
        --markets-folder simulator_ready_markets_liquidity_all_20260519_official_ref \
        --true-outcomes-csv simulator_ready_markets_liquidity_all_20260519_true_outcomes.csv
"""

import argparse
import csv as _csv
from pathlib import Path

import numpy as np
import pandas as pd

LEVELS = [0.02, 0.05, 0.10, 0.15]
MOM_WINDOWS_S = (15, 60, 120, 240)
VOL_WINDOWS_S = (60, 120)

FEATURES = (["deficit_pct"]
            + [f"mom_toward_{w}s" for w in MOM_WINDOWS_S]
            + [f"vol_{w}s_pct" for w in VOL_WINDOWS_S]
            + ["seconds_left"])

USECOLS = ["unix_time", "seconds_left", "up_price", "down_price",
           "btc_binance", "btc_chainlink", "price_to_beat"]


def load_truth(csv_path: str) -> dict:
    truth = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            outcome = (row.get("true_outcome") or "").strip().lower()
            if outcome in ("up", "down"):
                truth[row["slug"]] = outcome
    return truth


def market_touch_features(csv_path: Path, true_outcome: str,
                          min_sec: float, spacing_s: float) -> list[dict]:
    df = pd.read_csv(csv_path, usecols=USECOLS)
    btc = (df["btc_chainlink"].where(df["btc_chainlink"].notna(),
                                     df["btc_binance"]).ffill().to_numpy())
    ptb = df["price_to_beat"].dropna()
    if ptb.empty or np.all(np.isnan(btc)):
        return []
    strike = float(ptb.iloc[-1])
    times = df["unix_time"].to_numpy(dtype=float)
    seconds_left = df["seconds_left"].to_numpy()

    records = []
    for side in ("up", "down"):
        prices = df[f"{side}_price"].to_numpy()
        won = side == true_outcome
        sign = 1.0 if side == "up" else -1.0  # direction of BTC move side needs
        eligible = seconds_left >= min_sec
        for level in LEVELS:
            idx = np.flatnonzero(eligible & (prices > 0) & (prices <= level))
            if idx.size == 0:
                continue
            # every touch instance, at most one per spacing_s; the first one
            # is flagged is_first (the original first-touch analysis).
            last_t = -np.inf
            first = True
            for i0 in (int(i) for i in idx):
                if times[i0] - last_t < spacing_s:
                    continue
                last_t = times[i0]
                if np.isnan(btc[i0]):
                    continue
                p0 = float(prices[i0])
                t0 = times[i0]
                rec = {
                    "market": csv_path.stem, "side": side, "level": level,
                    "entry_price": p0, "seconds_left": float(seconds_left[i0]),
                    "won": won, "payoff": (1.0 / p0 - 1.0) if won else -1.0,
                    "is_first": first,
                    # signed: positive = BTC must move this % for side to win
                    "deficit_pct": sign * (strike - btc[i0]) / strike * 100.0,
                }
                first = False
                for w in MOM_WINDOWS_S:
                    j = int(np.searchsorted(times, t0 - w))
                    past = btc[j] if j < i0 else np.nan
                    rec[f"mom_toward_{w}s"] = (
                        sign * (btc[i0] - past) / strike * 100.0
                        if not np.isnan(past) else np.nan)
                for w in VOL_WINDOWS_S:
                    j = int(np.searchsorted(times, t0 - w))
                    win = btc[j:i0 + 1]
                    win = win[~np.isnan(win)]
                    rec[f"vol_{w}s_pct"] = (
                        (win.max() - win.min()) / strike * 100.0
                        if win.size >= 2 else np.nan)
                records.append(rec)
    return records


def rank_auc(values: np.ndarray, flips: np.ndarray) -> float:
    """AUC of feature vs flip via rank statistic. >0.5 = higher value -> flip."""
    ok = ~np.isnan(values)
    v, y = values[ok], flips[ok]
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = pd.Series(v).rank().to_numpy()
    return float((order[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def print_quartile_table(sub: pd.DataFrame, feature: str) -> None:
    d = sub.dropna(subset=[feature])
    if len(d) < 20:
        print(f"  {feature}: too few rows ({len(d)})")
        return
    try:
        bins = pd.qcut(d[feature], 4, duplicates="drop")
    except ValueError:
        print(f"  {feature}: degenerate distribution")
        return
    auc = rank_auc(d[feature].to_numpy(), d["won"].to_numpy())
    print(f"  {feature}  (AUC {auc:.2f})")
    grp = d.groupby(bins, observed=True)
    for interval, g in grp:
        flips = int(g["won"].sum())
        ev = g["payoff"].mean()
        flag = "  <-- EV+" if ev > 0 else ""
        print(f"    {str(interval):>22} n={len(g):>4} flips={flips:>3} "
              f"({100 * g['won'].mean():>4.1f}%) EV/$1={ev:>7.3f}{flag}")


def print_interaction(sub: pd.DataFrame, deficit_cut: float,
                      mom_col: str, per_market: bool = False) -> None:
    d = sub.dropna(subset=["deficit_pct", mom_col])
    if d.empty:
        return
    print(f"  deficit x {mom_col} 2x2 (near = |deficit| <= {deficit_cut}%):")
    for near in (True, False):
        for mom_pos in (True, False):
            g = d[(d["deficit_pct"].abs() <= deficit_cut) == near]
            g = g[(g[mom_col] > 0) == mom_pos]
            lbl = (f"{'near' if near else 'far '} & "
                   f"mom{'+' if mom_pos else '-'}")
            if g.empty:
                print(f"    {lbl}: n=0")
                continue
            ev = g["payoff"].mean()
            flag = "  <-- EV+" if ev > 0 else ""
            extra = ""
            if per_market:
                mk = g.groupby("market")["won"].first()
                extra = (f"  [markets={len(mk)} flip_mkts={int(mk.sum())}"
                         f" ({100 * mk.mean():.1f}%)]")
            print(f"    {lbl}: n={len(g):>4} flips={int(g['won'].sum()):>3} "
                  f"({100 * g['won'].mean():>4.1f}%) EV/$1={ev:>7.3f}"
                  f"{extra}{flag}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets-folder", required=True)
    ap.add_argument("--true-outcomes-csv", required=True)
    ap.add_argument("--min-seconds-left", type=float, default=5.0)
    ap.add_argument("--deficit-cut", type=float, default=0.05,
                    help="near-strike threshold for the 2x2 interaction")
    ap.add_argument("--spacing", type=float, default=10.0,
                    help="min seconds between touch instances per side/level")
    args = ap.parse_args()

    truth = load_truth(args.true_outcomes_csv)
    folder = Path(args.markets_folder)
    records, skipped = [], 0
    for path in sorted(folder.glob("*.csv")):
        outcome = truth.get(path.stem)
        if outcome is None:
            skipped += 1
            continue
        records.extend(market_touch_features(
            path, outcome, args.min_seconds_left, args.spacing))
    df = pd.DataFrame(records)
    df_first = df[df["is_first"]]
    print(f"touch instances: {len(df)} (first-touches: {len(df_first)}; "
          f"spacing {args.spacing}s; skipped {skipped} markets without truth)")
    print("flip = side truly won at resolution; payoff = 1/p-1 if flip else -1")

    for label, data, per_market in (("FIRST-TOUCH", df_first, False),
                                    ("ALL-INSTANCE", df, True)):
        for level in LEVELS:
            sub = data[data["level"] == level]
            if sub.empty:
                continue
            n, flips = len(sub), int(sub["won"].sum())
            print(f"\n{'=' * 74}\n{label} LEVEL <= {level}: n={n}, "
                  f"flips={flips} ({100 * flips / n:.1f}%), "
                  f"EV/$1={sub['payoff'].mean():.3f}, "
                  f"breakeven flip% = {100 * sub['entry_price'].mean():.1f}"
                  f"\n{'=' * 74}")
            for feature in FEATURES:
                print_quartile_table(sub, feature)
            print_interaction(sub, args.deficit_cut, "mom_toward_60s",
                              per_market=per_market)

    # AUC summary matrices
    for label, data in (("FIRST-TOUCH", df_first), ("ALL-INSTANCE", df)):
        print(f"\n{'=' * 74}\n{label} AUC summary (0.5 = no signal; >0.5 "
              f"higher value -> more flips)\n{'=' * 74}")
        print(f"{'feature':>18}"
              + "".join(f"{'<=' + str(l):>9}" for l in LEVELS))
        for feature in FEATURES:
            cells = []
            for level in LEVELS:
                sub = data[data["level"] == level]
                auc = rank_auc(sub[feature].to_numpy(), sub["won"].to_numpy())
                cells.append(f"{auc:>9.2f}" if not np.isnan(auc)
                             else f"{'-':>9}")
            print(f"{feature:>18}" + "".join(cells))

    # the flips themselves, with features at first touch, for eyeballing
    print("\nFLIPS at level <= 0.15 (features at first touch):")
    cols = ["market", "side", "entry_price", "seconds_left", "deficit_pct",
            "mom_toward_60s", "vol_60s_pct"]
    w = df_first[(df_first["level"] == 0.15)
                 & df_first["won"]].sort_values("deficit_pct")
    print(w[cols].to_string(index=False,
                            float_format=lambda x: f"{x:.3f}"))


if __name__ == "__main__":
    main()
