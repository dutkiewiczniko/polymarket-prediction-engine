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
    args = parser.parse_args()

    result = run_from_config(
        market_csv=args.market_csv,
        strategy_config=args.strategy_config,
        output_csv=args.output_csv,
        final_outcome=args.final_outcome,
    )

    print("Simulation complete:")
    for key, value in asdict(result).items():
        print(f"  {key}: {value}")


if __name__ == "__main__":
    main()
