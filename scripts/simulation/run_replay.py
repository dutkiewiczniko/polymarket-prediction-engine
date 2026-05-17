import argparse
import sys
from dataclasses import asdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.replay import run_from_config


def main():
    parser = argparse.ArgumentParser(description="Replay a recorded BTC up/down market through one strategy config.")
    parser.add_argument("--market-csv", required=True, help="Path to recorded data/btc-updown-5m-*.csv")
    parser.add_argument("--strategy-config", required=True, help="Path to configs/strategies/*.yaml")
    parser.add_argument("--output-csv", required=True, help="Where to write simulated trajectory CSV")
    parser.add_argument("--final-outcome", choices=["up", "down"], default=None, help="Optional manual final outcome override")
    parser.add_argument("--liquidity-aware-execution", action="store_true", help="Cap paper fills using logged orderbook depth columns.")
    parser.add_argument("--liquidity-depth-window-cents", type=int, default=2, help="Use visible depth within this many cents.")
    parser.add_argument("--liquidity-fill-fraction", type=float, default=1.0, help="Fraction of visible depth considered fillable.")
    parser.add_argument(
        "--liquidity-missing-depth-policy",
        choices=["skip", "allow"],
        default="skip",
        help="What to do when liquidity-aware execution is on but a row has no depth data.",
    )
    args = parser.parse_args()

    result = run_from_config(
        market_csv=args.market_csv,
        strategy_config=args.strategy_config,
        output_csv=args.output_csv,
        final_outcome=args.final_outcome,
        liquidity_aware_execution=args.liquidity_aware_execution,
        liquidity_depth_window_cents=args.liquidity_depth_window_cents,
        liquidity_fill_fraction=args.liquidity_fill_fraction,
        liquidity_missing_depth_policy=args.liquidity_missing_depth_policy,
    )

    print("Simulation complete:")
    for key, value in asdict(result).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
