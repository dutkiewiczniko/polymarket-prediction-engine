import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.batch import run_batch
from simulator.config_loader import load_yaml
from scripts.simulation.strategy_report import generate_report


DEFAULT_CONFIG = Path("configs/simulation_batch.yaml")


def parse_args():
    parser = argparse.ArgumentParser(description="Run simulator batch jobs.")
    parser.add_argument("--config", help=f"Batch config path. Default: {DEFAULT_CONFIG}")
    parser.add_argument("--markets-folder", help="Folder containing market CSV files.")
    parser.add_argument("--market-pattern", help="Market CSV glob pattern.")
    parser.add_argument("--output-root", help="Root folder where batch run folders are written.")
    parser.add_argument("--batch-id", help="Name for this run under output-root.")
    parser.add_argument("--max-markets", type=int, help="Limit how many market CSVs to simulate.")
    parser.add_argument(
        "--compound-balance",
        action="store_true",
        help="Carry each strategy's final balance into its next market as starting balance.",
    )
    parser.add_argument(
        "--no-compound-balance",
        action="store_true",
        help="Force non-compounding mode even if the config enables compounding.",
    )
    parser.add_argument(
        "--no-input",
        action="store_true",
        help="Do not ask interactive questions; use config values and any CLI overrides.",
    )
    return parser.parse_args()


def prompt_value(label, default_value):
    value = input(f"{label} [{default_value}]: ").strip()
    return value if value else default_value


def prompt_optional_int(label, default_value):
    default_display = "" if default_value is None else default_value
    value = input(f"{label} [{default_display}]: ").strip()
    if not value:
        return default_value
    return int(value)


def prompt_bool(label, default_value):
    default_text = "y" if default_value else "n"
    value = input(f"{label} [y/n, default {default_text}]: ").strip().lower()
    if not value:
        return default_value
    if value in {"y", "yes", "true", "1"}:
        return True
    if value in {"n", "no", "false", "0"}:
        return False
    raise ValueError(f"Invalid yes/no value: {value}")


def main():
    args = parse_args()
    interactive = not args.no_input and len(sys.argv) == 1

    print("BTC simulator batch runner")
    print()
    print("Market input comes from: config markets_folder, or --markets-folder")
    print("Run output goes to:      output_root/batch_id")
    print()

    config_path = Path(args.config) if args.config else DEFAULT_CONFIG
    if interactive:
        config_path = Path(prompt_value("Batch config path", config_path))

    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        return

    config = load_yaml(config_path)

    markets_folder = args.markets_folder
    market_pattern = args.market_pattern
    output_root = args.output_root
    batch_id = args.batch_id
    max_markets = args.max_markets
    compound_balance = None
    if args.compound_balance:
        compound_balance = True
    if args.no_compound_balance:
        compound_balance = False

    if interactive:
        markets_folder = prompt_value("Markets folder", markets_folder or config.get("markets_folder", "data"))
        market_pattern = prompt_value(
            "Market filename pattern",
            market_pattern or config.get("market_pattern", "btc-updown-5m-*.csv"),
        )
        output_root = prompt_value("Output root folder", output_root or config.get("output_root", "runs"))
        batch_id = prompt_value("Batch id", batch_id or config.get("batch_id", "batch_test"))
        max_markets = prompt_optional_int("Max markets, blank for all", max_markets or config.get("max_markets"))
        compound_balance = prompt_bool(
            "Compound balance across markets per strategy",
            bool(config.get("compound_balance", False)) if compound_balance is None else compound_balance,
        )

    summary_path = run_batch(
        config_path,
        markets_folder=markets_folder,
        market_pattern=market_pattern,
        output_root=output_root,
        batch_id=batch_id,
        max_markets=max_markets,
        compound_balance=compound_balance,
    )
    run_folder = Path(summary_path).parent
    try:
        print()
        print("Generating strategy report...")
        generate_report(run_folder=run_folder)
    except Exception as exc:
        print(f"Strategy report generation failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
