"""Comeback ("flip") frequency and EV as a function of entry price, scored
against TRUE Polymarket resolutions.

Generalises analyze_lottery_tickets_true_outcomes.py (which only looked at the
<=0.025 cheap zone) to a full grid of price levels. The question: when a side's
price drops to level L (the market considers that side nearly lost), how often
does that side still WIN ("flip"/"comeback"), and what is the EV of buying $1
of it at that moment?

DATA-INTEGRITY RULE (non-negotiable): outcomes come ONLY from the true-outcomes
CSV (real Polymarket resolutions). Outcomes are NEVER inferred from tick data --
the recorded strikes are mixed-source and inference is wrong in 42/320 markets
(see memory/btc_feed_reliability.md, memory/momentum_findings.md).

Two touch conventions, both reported:
  * FIRST-TOUCH (primary, a limit-order model): for each (market, side, level)
    take the FIRST tick where 0 < price <= L and seconds_left >= min_sec. One
    observation. Payoff = 1/entry_price - 1 if that side truly won, else -1.
    This models placing a resting limit buy at ~L and holding to resolution.
    Levels are cumulative thresholds (price <= L), so a lower level's touched
    set is a subset of a higher level's -- that is the correct semantics for
    independent limit orders at different prices.
  * ANY-TOUCH (secondary, matches the lottery-ticket sim): buy $1 every time
    price <= L, at most once per --spacing seconds. Overweights markets that
    dwell cheap; shown for comparison with the prior lottery work.

Breakeven: at entry price p, EV/$1 > 0 iff flip_rate > p. So the "breakeven
flip%" for a level is just its mean entry price.

Optional distance-to-strike (--ref-folder pointing at the *_chainlink_ref bench,
whose price_to_beat is corrected): for each first-touch, |BTC - strike|/strike%
at the touch tick, split near (<=0.05%) vs far.

Usage:
    python scripts/analysis/analyze_comeback_by_price_level_true_outcomes.py \
        --markets-folder simulator_ready_markets_liquidity_all_20260519 \
        --true-outcomes-csv simulator_ready_markets_liquidity_all_20260519_true_outcomes.csv \
        --ref-folder simulator_ready_markets_liquidity_all_20260519_chainlink_ref
"""

import argparse
import csv as _csv
from pathlib import Path

import numpy as np
import pandas as pd

# Full price grid. Below ~0.05 is the "nearly lost" zone the lottery work
# covered; 0.05-0.45 walks the entry price up toward the 50/50 coin flip.
LEVELS = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05, 0.075, 0.10, 0.15,
          0.20, 0.25, 0.30, 0.35, 0.40, 0.45]

# Coarse time-remaining buckets at the moment of touch. "Early cheap" (a side
# crashed with minutes left, lots of time to reverse) vs "late cheap" (crashed
# near resolution, little time) are very different animals.
SEC_BINS = [(5, 60, "late <60s"), (60, 180, "mid 60-180s"),
            (180, 300.01, "early >=180s")]

NEAR_STRIKE_PCT = 0.05  # |BTC-strike|/strike% threshold for "near strike"

# A "real comeback": the crashed side's price later climbs to >= this, a level
# you could actually sell into. Overridable with --comeback-threshold.
COMEBACK_THRESHOLD = 0.85


def load_truth(csv_path: str) -> dict:
    truth = {}
    with open(csv_path, newline="", encoding="utf-8") as f:
        for row in _csv.DictReader(f):
            outcome = (row.get("true_outcome") or "").strip().lower()
            if outcome in ("up", "down"):
                truth[row["slug"]] = outcome
    return truth


def market_touches(csv_path: Path, true_outcome: str, min_sec: float,
                   spacing_s: float, ref_lookup: dict | None) -> list[dict]:
    """First-touch and any-touch records for every side x level in one market."""
    df = pd.read_csv(csv_path,
                     usecols=["unix_time", "seconds_left", "up_price", "down_price"])
    times = df["unix_time"].to_numpy()
    seconds_left = df["seconds_left"].to_numpy()

    records = []
    for side in ("up", "down"):
        prices = df[f"{side}_price"].to_numpy()
        won = side == true_outcome
        eligible = seconds_left >= min_sec
        for level in LEVELS:
            hit = eligible & (prices > 0) & (prices <= level)
            idx = np.flatnonzero(hit)
            if idx.size == 0:
                continue
            # --- first touch (one observation) ---
            i0 = idx[0]
            p0 = float(prices[i0])
            # Did the side stage a real comeback -- price later climbs to
            # >= COMEBACK_THRESHOLD, a level you could sell into (vs. merely
            # scraping a win at resolution)? ev_sell models selling at the
            # threshold if reached, else holding to resolution.
            future_max = float(np.nanmax(prices[i0:])) if prices[i0:].size else p0
            reached = future_max >= COMEBACK_THRESHOLD
            ev_sell = ((COMEBACK_THRESHOLD / p0 - 1.0) if reached
                       else ((1.0 / p0 - 1.0) if won else -1.0))
            rec = {
                "market": csv_path.stem, "side": side, "level": level,
                "entry_price": p0, "seconds_left": float(seconds_left[i0]),
                "won": won, "payoff": (1.0 / p0 - 1.0) if won else -1.0,
                "reached": reached, "ev_sell": ev_sell, "future_max": future_max,
            }
            if ref_lookup is not None:
                rec["dist_pct"] = ref_lookup.get(int(times[i0]), np.nan)
            records.append(rec)
            # --- any-touch spaced tickets (secondary) ---
            last_t = -np.inf
            n_tickets, ticket_ev = 0, 0.0
            for j in idx:
                if times[j] - last_t < spacing_s:
                    continue
                last_t = times[j]
                pj = float(prices[j])
                n_tickets += 1
                ticket_ev += (1.0 / pj - 1.0) if won else -1.0
            rec["n_tickets"] = n_tickets
            rec["ticket_ev_sum"] = ticket_ev
    return records


def build_ref_lookup(ref_folder: Path, stem: str) -> dict | None:
    """unix_time -> |BTC-strike|/strike% from the corrected chainlink_ref bench."""
    path = ref_folder / f"{stem}.csv"
    if not path.exists():
        return None
    r = pd.read_csv(path, usecols=["unix_time", "btc_chainlink", "btc_binance",
                                    "price_to_beat"])
    btc = r["btc_chainlink"].where(r["btc_chainlink"].notna(), r["btc_binance"])
    ptb = r["price_to_beat"].dropna()
    if btc.dropna().empty or ptb.empty:
        return None
    strike = float(ptb.iloc[-1])
    dist = (btc - strike).abs() / strike * 100.0
    return dict(zip(r["unix_time"].astype(int), dist))


def fmt_level(level: float) -> str:
    return f"{level:.3f}".rstrip("0").rstrip(".")


def print_first_touch_table(df: pd.DataFrame, title: str) -> None:
    print(f"\n=== {title} ===")
    print(f"{'level':>7} {'n':>5} {'flips':>6} {'flip%':>6} {'brkeven%':>8} "
          f"{'mean_p':>7} {'EV/$1':>7} {'med_sec':>8}")
    for level in LEVELS:
        sub = df[df["level"] == level]
        if sub.empty:
            continue
        n = len(sub)
        flips = int(sub["won"].sum())
        flip_pct = 100.0 * flips / n
        mean_p = sub["entry_price"].mean()
        ev = sub["payoff"].mean()
        med_sec = sub["seconds_left"].median()
        flag = "  <-- EV+" if ev > 0 else ""
        print(f"{fmt_level(level):>7} {n:>5} {flips:>6} {flip_pct:>5.1f}% "
              f"{100 * mean_p:>7.1f} {mean_p:>7.3f} {ev:>7.3f} {med_sec:>8.0f}{flag}")


def print_comeback_table(df: pd.DataFrame, title: str) -> None:
    """Flip = price later reached COMEBACK_THRESHOLD (a sellable comeback)."""
    print(f"\n=== {title} (comeback = price later >= {COMEBACK_THRESHOLD}, "
          f"sell there; else hold to resolution) ===")
    print(f"{'level':>7} {'n':>5} {'reached':>8} {'reach%':>7} {'brkeven%':>8} "
          f"{'mean_p':>7} {'EV/$1':>7} {'med_sec':>8}")
    for level in LEVELS:
        sub = df[df["level"] == level]
        if sub.empty:
            continue
        n = len(sub)
        reached = int(sub["reached"].sum())
        mean_p = sub["entry_price"].mean()
        brkeven = 100 * mean_p / COMEBACK_THRESHOLD  # need reach% > p/threshold
        ev = sub["ev_sell"].mean()
        med_sec = sub["seconds_left"].median()
        flag = "  <-- EV+" if ev > 0 else ""
        print(f"{fmt_level(level):>7} {n:>5} {reached:>8} "
              f"{100 * reached / n:>6.1f}% {brkeven:>7.1f} {mean_p:>7.3f} "
              f"{ev:>7.3f} {med_sec:>8.0f}{flag}")


def ev_at_threshold(sub: pd.DataFrame, thr: float) -> float:
    """EV/$1 of a take-profit at price `thr`: sell there if the side ever
    reaches it after entry, else hold to resolution."""
    p = sub["entry_price"].to_numpy()
    reached = sub["future_max"].to_numpy() >= thr
    won = sub["won"].to_numpy()
    payoff = np.where(reached, thr / p - 1.0,
                      np.where(won, 1.0 / p - 1.0, -1.0))
    return float(payoff.mean())


def print_threshold_sweep(df: pd.DataFrame) -> None:
    """EV/$1 as a function of take-profit threshold, per entry level.

    Shape reading: EV rising with threshold => prices trend (hold longer wins);
    EV falling => prices mean-revert (take profit early wins); flat => calibrated
    martingale, exit rule is irrelevant. T=1.00 == hold to resolution.
    """
    thrs = [0.80, 0.85, 0.90, 0.92, 0.95, 0.97, 0.99, 1.00]
    print("\n=== take-profit threshold sweep: EV/$1 by (entry level x sell "
          "threshold) ===")
    print("  (best threshold per row marked *; T=1.00 = hold to resolution)")
    print(f"{'level':>7}" + "".join(f"{('T'+f'{t:.2f}'):>8}" for t in thrs)
          + f"{'best':>7}")
    for level in LEVELS:
        sub = df[df["level"] == level]
        if sub.empty:
            continue
        evs = [ev_at_threshold(sub, t) for t in thrs]
        best_i = int(np.argmax(evs))
        cells = "".join((f"{ev:>7.3f}" + ("*" if i == best_i else " "))
                        for i, ev in enumerate(evs))
        print(f"{fmt_level(level):>7}{cells}{('T'+f'{thrs[best_i]:.2f}'):>7}")
    # Pooled buckets: cheap moonshot zone vs the mid/rich zone, to see the
    # aggregate best exit for each (they behave oppositely).
    print("-" * 78)
    for lbl, mask in [("<=0.05", df["level"] <= 0.05),
                      (">=0.10", df["level"] >= 0.10),
                      ("ALL", df["level"].notna())]:
        sub = df[mask]
        evs = [ev_at_threshold(sub, t) for t in thrs]
        best_i = int(np.argmax(evs))
        cells = "".join((f"{ev:>7.3f}" + ("*" if i == best_i else " "))
                        for i, ev in enumerate(evs))
        print(f"{lbl:>7}{cells}{('T'+f'{thrs[best_i]:.2f}'):>7}")


def print_sec_bins(df: pd.DataFrame) -> None:
    print("\n=== first-touch flip% by seconds_left at touch (combined sides) ===")
    hdr = "".join(f"{lbl:>18}" for _, _, lbl in SEC_BINS)
    print(f"{'level':>7}{hdr}")
    for level in LEVELS:
        lvl = df[df["level"] == level]
        if lvl.empty:
            continue
        cells = []
        for lo, hi, _ in SEC_BINS:
            b = lvl[(lvl["seconds_left"] >= lo) & (lvl["seconds_left"] < hi)]
            if b.empty:
                cells.append(f"{'-':>18}")
            else:
                cells.append(f"{int(b['won'].sum())}/{len(b)} "
                             f"{100 * b['won'].mean():.0f}% ev{b['payoff'].mean():+.2f}"
                             .rjust(18))
        print(f"{fmt_level(level):>7}" + "".join(cells))


def print_distance_split(df: pd.DataFrame) -> None:
    if "dist_pct" not in df.columns or df["dist_pct"].dropna().empty:
        return
    d = df.dropna(subset=["dist_pct"])
    print(f"\n=== first-touch flip% by distance-to-strike (near = |BTC-strike| "
          f"<= {NEAR_STRIKE_PCT}%) ===")
    print(f"{'level':>7} {'near n':>7} {'near flip%':>11} {'near EV':>8} "
          f"{'far n':>6} {'far flip%':>10} {'far EV':>7}")
    for level in LEVELS:
        lvl = d[d["level"] == level]
        if lvl.empty:
            continue
        near = lvl[lvl["dist_pct"] <= NEAR_STRIKE_PCT]
        far = lvl[lvl["dist_pct"] > NEAR_STRIKE_PCT]
        nn = (f"{len(near)}" if len(near) else "0")
        nf = (f"{100 * near['won'].mean():.1f}%" if len(near) else "-")
        ne = (f"{near['payoff'].mean():+.3f}" if len(near) else "-")
        fn = (f"{len(far)}" if len(far) else "0")
        ff = (f"{100 * far['won'].mean():.1f}%" if len(far) else "-")
        fe = (f"{far['payoff'].mean():+.3f}" if len(far) else "-")
        print(f"{fmt_level(level):>7} {nn:>7} {nf:>11} {ne:>8} "
              f"{fn:>6} {ff:>10} {fe:>7}")


def print_any_touch_table(df: pd.DataFrame) -> None:
    print("\n=== any-touch spaced-ticket EV (buy $1 per touch, spacing applied) ===")
    print(f"{'level':>7} {'tickets':>8} {'wins':>6} {'win%':>6} {'EV/$1':>7}")
    for level in LEVELS:
        sub = df[df["level"] == level]
        if sub.empty:
            continue
        n = int(sub["n_tickets"].sum())
        if n == 0:
            continue
        ev_sum = sub["ticket_ev_sum"].sum()
        win_tickets = int(sub.loc[sub["won"], "n_tickets"].sum())
        print(f"{fmt_level(level):>7} {n:>8} {win_tickets:>6} "
              f"{100 * win_tickets / n:>5.1f}% {ev_sum / n:>7.3f}")


def main():
    global COMEBACK_THRESHOLD
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets-folder", required=True)
    ap.add_argument("--true-outcomes-csv", required=True)
    ap.add_argument("--ref-folder", default=None,
                    help="corrected *_chainlink_ref bench for distance-to-strike")
    ap.add_argument("--min-seconds-left", type=float, default=5.0)
    ap.add_argument("--spacing", type=float, default=10.0,
                    help="min seconds between any-touch tickets")
    ap.add_argument("--comeback-threshold", type=float, default=COMEBACK_THRESHOLD,
                    help="price a crashed side must later reach to count as a "
                         "sellable comeback (default 0.85)")
    args = ap.parse_args()
    COMEBACK_THRESHOLD = args.comeback_threshold

    truth = load_truth(args.true_outcomes_csv)
    folder = Path(args.markets_folder)
    ref_folder = Path(args.ref_folder) if args.ref_folder else None

    records, skipped, n_markets = [], 0, 0
    for path in sorted(folder.glob("*.csv")):
        outcome = truth.get(path.stem)
        if outcome is None:
            skipped += 1
            continue
        n_markets += 1
        ref_lookup = build_ref_lookup(ref_folder, path.stem) if ref_folder else None
        records.extend(market_touches(
            path, outcome, args.min_seconds_left, args.spacing, ref_lookup))

    df = pd.DataFrame(records)
    truth_in = {p.stem: truth[p.stem] for p in folder.glob("*.csv")
                if p.stem in truth}
    up_rate = 100.0 * sum(v == "up" for v in truth_in.values()) / max(len(truth_in), 1)
    print(f"markets scored: {n_markets} (skipped {skipped} without truth); "
          f"base rate up={up_rate:.1f}% down={100 - up_rate:.1f}%")
    print(f"min_seconds_left={args.min_seconds_left}  any-touch spacing={args.spacing}s")
    print("payoff = 1/entry_price - 1 if that side truly WON, else -1. "
          "EV>0 iff flip% > breakeven% (= mean entry price).")
    if df.empty:
        print("no touches recorded.")
        return

    print_first_touch_table(df, "FIRST-TOUCH combined (up + down sides)")
    print_first_touch_table(df[df["side"] == "down"], "FIRST-TOUCH down side only")
    print_first_touch_table(df[df["side"] == "up"], "FIRST-TOUCH up side only")
    print_comeback_table(df, "COMEBACK combined (up + down sides)")
    print_threshold_sweep(df)
    print_sec_bins(df)
    print_distance_split(df)
    print_any_touch_table(df)

    # Flip markets at the higher levels -- the actual comebacks worth eyeballing.
    print("\n=== flip markets (side truly won after first-touching level) ===")
    for level in [0.05, 0.10, 0.15, 0.20, 0.30]:
        w = df[(df["level"] == level) & (df["won"])]
        if w.empty:
            continue
        items = sorted(f"{r.market}[{r.side}@{r.entry_price:.3f},"
                       f"{r.seconds_left:.0f}s]" for r in w.itertuples())
        print(f"level {fmt_level(level)}: {len(items)} flips")
        for it in items:
            print(f"    {it}")


if __name__ == "__main__":
    main()
