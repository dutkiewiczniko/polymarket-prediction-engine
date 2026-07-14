"""Where do cheap lottery tickets pay, and can we filter the dead ones?

The cheap-lottery leg (buy a side at <= 0.025) drives most of the strategy
family's P&L: near-zero median in calm regimes, huge convex wins in whipsaw
stretches (see OOS verdict in memory). This script simulates every ticket
opportunity across a market folder and buckets ticket EV by features that are
observable AT PURCHASE TIME and already exist as rule metrics:

- abs_btc_distance_to_price_to_beat_pct  (how far BTC is from the strike)
- seconds_left
- recent BTC movement (60s range, pct) -- volatility proxy

Ticket simulation: for each side, while its price is <= --max-price and
seconds_left >= --min-seconds-left, buy $1 at the quoted price at most once per
--ticket-spacing-s seconds. Payoff per ticket = (1/price - 1) if that side wins
else -1.

Usage:
    python scripts/analysis/analyze_lottery_tickets.py \
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

USECOLS = [
    "unix_time", "seconds_left", "up_price", "down_price",
    "btc_binance", "btc_chainlink", "price_to_beat",
]


def market_tickets(
    csv_path: Path,
    max_price: float,
    min_seconds_left: float,
    spacing_s: float,
    true_outcomes: dict[str, str] | None = None,
) -> list[dict]:
    df = pd.read_csv(csv_path, usecols=USECOLS)
    btc = df["btc_chainlink"].where(df["btc_chainlink"].notna(), df["btc_binance"])
    df = df.assign(btc=btc)
    ptb = df["price_to_beat"].dropna()
    if df["btc"].dropna().empty or ptb.empty:
        return []
    strike = ptb.iloc[-1]
    if true_outcomes is not None:
        # Tick inference got 42/320 outcomes wrong on the mixed-strike bench;
        # skip any market not covered by the truth CSV rather than guess.
        outcome = true_outcomes.get(csv_path.stem)
        if outcome not in ("up", "down"):
            return []
    else:
        outcome = "up" if df["btc"].dropna().iloc[-1] >= strike else "down"

    times = df["unix_time"].to_numpy()
    btc_vals = df["btc"].to_numpy()

    tickets = []
    for side in ("up", "down"):
        prices = df[f"{side}_price"].to_numpy()
        seconds_left = df["seconds_left"].to_numpy()
        last_buy_t = -np.inf
        for i in range(len(df)):
            price = prices[i]
            if not (0 < price <= max_price):
                continue
            if seconds_left[i] < min_seconds_left:
                continue
            if times[i] - last_buy_t < spacing_s:
                continue
            last_buy_t = times[i]
            here_btc = btc_vals[i]
            if np.isnan(here_btc):
                continue
            # 60s realized BTC range ending at this tick (volatility proxy)
            window = btc_vals[(times >= times[i] - 60) & (times <= times[i])]
            window = window[~np.isnan(window)]
            range_pct = (window.max() - window.min()) / here_btc * 100 if len(window) > 1 else np.nan
            # Per-source signed distances: resolution is Chainlink but Binance
            # leads it -- a cheap ticket whose side Binance has ALREADY crossed
            # toward is a look-ahead buy (mission lead #3).
            bin_px = df["btc_binance"].iloc[i]
            cl_px = df["btc_chainlink"].iloc[i]
            sign = 1.0 if side == "up" else -1.0
            tickets.append({
                "market_file": csv_path.name,
                "side": side,
                "price": price,
                "seconds_left": seconds_left[i],
                "abs_dist_pct": abs(here_btc - strike) / strike * 100,
                "btc_range_60s_pct": range_pct,
                # >0 means that source is already on the bought side's winning
                # side of the strike (up: above, down: below)
                "binance_side_dist_pct": sign * (bin_px - strike) / strike * 100 if not np.isnan(bin_px) else np.nan,
                "chainlink_side_dist_pct": sign * (cl_px - strike) / strike * 100 if not np.isnan(cl_px) else np.nan,
                "won": side == outcome,
                "payoff": (1.0 / price - 1.0) if side == outcome else -1.0,
            })
    return tickets


def bucket_report(t: pd.DataFrame, col: str, edges: list[float]):
    t = t[t[col].notna()].copy()
    t["bucket"] = pd.cut(t[col], edges)
    g = t.groupby("bucket", observed=True).agg(
        tickets=("payoff", "size"),
        win_rate=("won", "mean"),
        ev_per_ticket=("payoff", "mean"),
        total_payoff=("payoff", "sum"),
    )
    print(f"\n=== ticket EV by {col} ===")
    print(g.round(3).to_string())


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--markets-folder", default="simulator_ready_markets_liquidity_all_20260519")
    parser.add_argument("--market-pattern", default="btc-updown-5m-*.csv")
    parser.add_argument("--max-price", type=float, default=0.025)
    parser.add_argument("--min-seconds-left", type=float, default=10.0)
    parser.add_argument("--ticket-spacing-s", type=float, default=15.0)
    parser.add_argument(
        "--true-outcomes-csv",
        default=None,
        help="CSV with slug,true_outcome columns (see fetch_true_outcomes.py). "
        "Markets listed resolve to the actual Polymarket outcome; markets NOT "
        "listed are skipped entirely (tick inference is untrustworthy).",
    )
    args = parser.parse_args()

    true_outcomes = None
    if args.true_outcomes_csv:
        truth = pd.read_csv(args.true_outcomes_csv)
        true_outcomes = dict(zip(truth["slug"], truth["true_outcome"]))
        print(f"True outcomes: {len(true_outcomes)} markets from {args.true_outcomes_csv}")

    folder = Path(args.markets_folder)
    files = sorted(folder.glob(args.market_pattern))
    all_tickets = []
    for f in files:
        all_tickets.extend(
            market_tickets(f, args.max_price, args.min_seconds_left, args.ticket_spacing_s, true_outcomes)
        )
    t = pd.DataFrame(all_tickets)
    if t.empty:
        raise SystemExit("No tickets found.")

    n_markets_with = t["market_file"].nunique()
    print(f"{len(t)} tickets across {n_markets_with}/{len(files)} markets | "
          f"win rate {t['won'].mean():.3f} | EV/ticket {t['payoff'].mean():+.3f} | "
          f"total {t['payoff'].sum():+.1f}")

    bucket_report(t, "abs_dist_pct", [0, 0.02, 0.05, 0.1, 0.2, 0.5, 10])
    bucket_report(t, "seconds_left", [10, 30, 60, 120, 200, 300])
    bucket_report(t, "btc_range_60s_pct", [0, 0.05, 0.1, 0.2, 0.5, 10])

    # Joint filter demo: the candidate rule condition
    for dist_max in (0.05, 0.1, 0.2):
        kept = t[t["abs_dist_pct"] <= dist_max]
        cut = t[t["abs_dist_pct"] > dist_max]
        print(
            f"\nfilter abs_dist_pct <= {dist_max}: keeps {len(kept)}/{len(t)} tickets, "
            f"kept EV {kept['payoff'].mean():+.3f} (total {kept['payoff'].sum():+.1f}), "
            f"cut EV {cut['payoff'].mean():+.3f} (total {cut['payoff'].sum():+.1f})"
        )

    out = folder.parent / f"{folder.name}_lottery_tickets.csv"
    t.to_csv(out, index=False)
    print(f"\nPer-ticket table: {out}")


if __name__ == "__main__":
    main()
