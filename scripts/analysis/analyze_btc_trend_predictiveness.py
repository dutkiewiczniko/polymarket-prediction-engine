"""Calibrate "strong trend" and "cheap entry" for macro BTC trend strategies.

Reads the btc_trend manifest produced by scripts/data/enrich_markets_with_btc_trend.py
plus each market CSV's outcome, then reports per lookback period:

- distribution of the trend values across markets
- how often the market resolves in the trend's direction (sign alignment),
  overall and bucketed by trend magnitude quartile
- how cheap the trend-favored side got during the market (for calibrating
  cheap-add rules)

Usage:
    python scripts/analysis/analyze_btc_trend_predictiveness.py \
        --markets-folder simulator_ready_markets_liquidity_all_20260519
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def market_outcome_and_prices(csv_path: Path) -> dict | None:
    df = pd.read_csv(
        csv_path,
        usecols=["seconds_left", "up_price", "down_price", "btc_binance", "btc_chainlink", "price_to_beat"],
    )
    btc = df["btc_chainlink"].where(df["btc_chainlink"].notna(), df["btc_binance"]).dropna()
    ptb = df["price_to_beat"].dropna()
    if btc.empty or ptb.empty:
        return None
    outcome = "up" if btc.iloc[-1] >= ptb.iloc[-1] else "down"
    # Entry prices while the market still has meaningful time left.
    live = df[df["seconds_left"] > 30]
    return {
        "outcome": outcome,
        "min_up_price": live["up_price"].min(),
        "min_down_price": live["down_price"].min(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--markets-folder", default="simulator_ready_markets_liquidity_all_20260519")
    parser.add_argument("--manifest", default=None, help="Defaults to <markets-folder>_btc_trend_manifest.csv")
    args = parser.parse_args()

    folder = Path(args.markets_folder)
    manifest_path = Path(args.manifest) if args.manifest else folder.parent / f"{folder.name}_btc_trend_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    trend_cols = [c for c in manifest.columns if c.startswith("btc_change_")]

    rows = []
    for record in manifest.itertuples(index=False):
        info = market_outcome_and_prices(folder / record.market_file)
        if info is None:
            continue
        rows.append({**record._asdict(), **info})
    df = pd.DataFrame(rows)

    up_rate = (df["outcome"] == "up").mean()
    print(f"{len(df)} markets | base rate: up {up_rate:.1%} / down {1 - up_rate:.1%}\n")

    for col in trend_cols:
        values = df[col].dropna()
        sub = df[df[col].notna()].copy()
        favored = sub[col].apply(lambda v: "up" if v > 0 else "down")
        aligned = (favored == sub["outcome"])
        print(f"=== {col} ===")
        print(
            "  distribution: "
            + " ".join(f"p{int(q*100)}={values.quantile(q):+.3f}" for q in (0.05, 0.25, 0.5, 0.75, 0.95))
        )
        print(f"  sign alignment (favored side wins): {aligned.mean():.1%}  (n={len(sub)})")

        sub["abs_mag"] = sub[col].abs()
        sub["mag_bucket"] = pd.qcut(sub["abs_mag"], 4, labels=["q1_weak", "q2", "q3", "q4_strong"], duplicates="drop")
        bucket = sub.groupby("mag_bucket", observed=True).apply(
            lambda g: pd.Series({
                "n": len(g),
                "mag_min": g["abs_mag"].min(),
                "mag_max": g["abs_mag"].max(),
                "favored_win_rate": (g[col].apply(lambda v: "up" if v > 0 else "down") == g["outcome"]).mean(),
                "favored_min_price_med": g.apply(
                    lambda r: r["min_up_price"] if r[col] > 0 else r["min_down_price"], axis=1
                ).median(),
            }),
            include_groups=False,
        )
        print(bucket.round(3).to_string())
        print()

    out_path = folder.parent / f"{folder.name}_btc_trend_outcomes.csv"
    df.to_csv(out_path, index=False)
    print(f"Per-market table: {out_path}")


if __name__ == "__main__":
    main()
