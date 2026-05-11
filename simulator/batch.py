import csv
import re
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from simulator.config_loader import (
    build_strategy_from_config,
    clone_strategy_config_with_seed,
    load_yaml,
)
from simulator.replay import run_simulation


def safe_name(value: str) -> str:
    """Convert a name into something safe for filenames."""
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "unnamed"


def discover_market_csvs(markets_folder: str | Path, pattern: str = "btc-updown-5m-*.csv") -> list[Path]:
    folder = Path(markets_folder)
    if not folder.exists():
        raise FileNotFoundError(f"markets_folder does not exist: {folder}")

    files = sorted(folder.glob(pattern))

    if not files:
        raise FileNotFoundError(f"No market CSV files found in {folder} using pattern {pattern!r}")

    return files


def resolve_effective_market_balance(master_balance: float, batch_cfg: dict) -> float | None:
    pct = batch_cfg.get("effective_market_balance_pct")
    if pct is not None:
        simulated_balance = master_balance * float(pct)
        min_balance = batch_cfg.get("effective_market_balance_min")
        max_balance = batch_cfg.get("effective_market_balance_max")
        if min_balance is not None:
            simulated_balance = max(simulated_balance, float(min_balance))
        if max_balance is not None:
            simulated_balance = min(simulated_balance, float(max_balance))
        return min(master_balance, simulated_balance)

    bands = batch_cfg.get("effective_market_balance_bands")
    if not bands:
        return None

    for band in bands:
        min_balance = band.get("min_total_balance")
        max_balance = band.get("max_total_balance")
        simulated_balance = band.get("simulated_balance")
        if simulated_balance is None:
            continue

        min_ok = min_balance is None or master_balance >= float(min_balance)
        max_ok = max_balance is None or master_balance < float(max_balance)
        if min_ok and max_ok:
            return min(master_balance, float(simulated_balance))

    return min(master_balance, float(batch_cfg.get("starting_balance", 100.0)))


def expand_strategy_runs(batch_cfg: dict) -> list[dict]:
    """Expand strategy entries into individual runnable configs.

    Example:
        random_basic with count: 2 becomes:
            random_basic__001
            random_basic__002
    """

    expanded = []

    for item in batch_cfg.get("strategies", []):
        config_path = item.get("config")
        count = int(item.get("count", 1))
        label = item.get("label")

        if not config_path:
            raise ValueError("Each strategy entry must include a 'config' path.")

        strategy_cfg = load_yaml(config_path)
        base_name = label or strategy_cfg.get("name") or Path(config_path).stem

        seed_start = item.get("seed_start", strategy_cfg.get("params", {}).get("seed"))
        seed_start = int(seed_start) if seed_start is not None else None

        for i in range(count):
            run_cfg = clone_strategy_config_with_seed(
                strategy_cfg,
                seed=(seed_start + i) if seed_start is not None else None,
            )

            if count > 1:
                run_name = f"{base_name}__{i + 1:03d}"
            else:
                run_name = base_name

            run_cfg["name"] = run_name

            expanded.append({
                "name": run_name,
                "source_config": str(config_path),
                "config": run_cfg,
                "run_index": i + 1,
            })

    if not expanded:
        raise ValueError("No strategies listed in batch config.")

    return expanded


def run_batch(
    batch_config_path: str | Path = "configs/simulation_batch.yaml",
    *,
    markets_folder: str | Path | None = None,
    market_pattern: str | None = None,
    output_root: str | Path | None = None,
    batch_id: str | None = None,
    max_markets: int | None = None,
    compound_balance: bool | None = None,
) -> Path:
    """Run many strategies across many market CSVs.

    Output structure:

        runs/<batch_id>/
          summary.csv
          trajectories/
            <strategy_name>/
              <market_slug>.csv
    """

    batch_config_path = Path(batch_config_path)
    batch_cfg = load_yaml(batch_config_path)

    batch_id = batch_id or batch_cfg.get("batch_id")
    if not batch_id:
        batch_id = datetime.now().strftime("batch_%Y%m%d_%H%M%S")

    batch_id = safe_name(batch_id)

    markets_folder = Path(markets_folder or batch_cfg.get("markets_folder", "data"))
    market_pattern = market_pattern or batch_cfg.get("market_pattern", "btc-updown-5m-*.csv")
    output_root = Path(output_root or batch_cfg.get("output_root", "runs"))

    starting_balance_default = float(batch_cfg.get("starting_balance", 100.0))
    order_usd_default = float(batch_cfg.get("order_usd", 1.0))
    final_outcome = batch_cfg.get("final_outcome")
    max_markets = max_markets if max_markets is not None else batch_cfg.get("max_markets")
    compound_balance = bool(
        batch_cfg.get("compound_balance", False) if compound_balance is None else compound_balance
    )

    markets = discover_market_csvs(markets_folder, market_pattern)
    if max_markets is not None:
        markets = markets[:int(max_markets)]

    strategy_runs = expand_strategy_runs(batch_cfg)

    batch_dir = output_root / batch_id
    trajectories_dir = batch_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    summary_path = batch_dir / "summary.csv"
    summary_fieldnames = [
        "status",
        "job_no",
        "total_jobs",
        "market_file",
        "strategy_name",
        "final_outcome",
        "final_balance",
        "starting_balance",
        "total_reward",
        "rows_written",
        "market_path",
        "strategy_config",
        "strategy_run_name",
        "output_csv",
        "compound_balance_mode",
        "strategy_market_no",
        "strategy_balance_before_market",
        "strategy_balance_after_market",
        "market_simulated_starting_balance",
        "market_simulated_final_balance",
        "error_type",
        "error_message",
    ]

    total_jobs = len(markets) * len(strategy_runs)
    job_no = 0
    completed_jobs = 0
    failed_jobs = 0
    strategy_balances = {
        safe_name(strategy_entry["name"]): float(strategy_entry["config"].get("starting_balance", starting_balance_default))
        for strategy_entry in strategy_runs
    }
    strategy_market_counts = {safe_name(strategy_entry["name"]): 0 for strategy_entry in strategy_runs}

    print(f"Batch: {batch_id}")
    print(f"Markets folder: {markets_folder}")
    print(f"Market pattern: {market_pattern}")
    print(f"Output folder: {batch_dir}")
    print(f"Markets: {len(markets)}")
    print(f"Strategy runs: {len(strategy_runs)}")
    print(f"Total simulations: {total_jobs}")
    print(f"Compound balances: {compound_balance}")
    print()

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=summary_fieldnames)
        writer.writeheader()
        f.flush()

        for market_path in markets:
            market_slug = safe_name(market_path.stem)

            for strategy_entry in strategy_runs:
                job_no += 1

                strategy_cfg = strategy_entry["config"]
                base_starting_balance = float(strategy_cfg.get("starting_balance", starting_balance_default))
                order_usd = float(strategy_cfg.get("order_usd", order_usd_default))

                strategy_name = safe_name(strategy_entry["name"])
                master_balance_before_market = (
                    strategy_balances[strategy_name] if compound_balance else base_starting_balance
                )
                effective_market_balance = resolve_effective_market_balance(master_balance_before_market, batch_cfg)
                starting_balance = (
                    effective_market_balance if effective_market_balance is not None else master_balance_before_market
                )
                strategy_market_counts[strategy_name] += 1
                strategy_market_no = strategy_market_counts[strategy_name]
                strategy_dir = trajectories_dir / strategy_name
                strategy_dir.mkdir(parents=True, exist_ok=True)
                output_csv = strategy_dir / f"{market_slug}.csv"

                print(f"[{job_no}/{total_jobs}] {market_path.name} -> {strategy_name}")

                try:
                    strategy = build_strategy_from_config(strategy_cfg)
                    result = run_simulation(
                        market_csv=market_path,
                        strategy=strategy,
                        output_csv=output_csv,
                        starting_balance=starting_balance,
                        order_usd=order_usd,
                        final_outcome=final_outcome,
                    )

                    result_dict = asdict(result)
                    summary_row = {
                        "status": "ok",
                        "job_no": job_no,
                        "total_jobs": total_jobs,
                        "market_path": str(market_path),
                        "strategy_config": strategy_entry["source_config"],
                        "strategy_run_name": strategy_name,
                        "output_csv": str(output_csv),
                        "compound_balance_mode": str(compound_balance).lower(),
                        "strategy_market_no": strategy_market_no,
                        "strategy_balance_before_market": master_balance_before_market,
                        "strategy_balance_after_market": (
                            master_balance_before_market + result.total_reward
                            if compound_balance and effective_market_balance is not None
                            else result.final_balance
                        ),
                        "market_simulated_starting_balance": starting_balance,
                        "market_simulated_final_balance": result.final_balance,
                        "error_type": "",
                        "error_message": "",
                        **result_dict,
                    }
                    if compound_balance:
                        strategy_balances[strategy_name] = float(
                            master_balance_before_market + result.total_reward
                            if effective_market_balance is not None
                            else result.final_balance
                        )
                    completed_jobs += 1
                except Exception as e:
                    summary_row = {
                        "status": "failed",
                        "job_no": job_no,
                        "total_jobs": total_jobs,
                        "market_file": str(market_path),
                        "strategy_name": strategy_name,
                        "final_outcome": "",
                        "final_balance": "",
                        "starting_balance": starting_balance,
                        "total_reward": "",
                        "rows_written": "",
                        "market_path": str(market_path),
                        "strategy_config": strategy_entry["source_config"],
                        "strategy_run_name": strategy_name,
                        "output_csv": str(output_csv),
                        "compound_balance_mode": str(compound_balance).lower(),
                        "strategy_market_no": strategy_market_no,
                        "strategy_balance_before_market": master_balance_before_market,
                        "strategy_balance_after_market": master_balance_before_market if compound_balance else "",
                        "market_simulated_starting_balance": starting_balance,
                        "market_simulated_final_balance": "",
                        "error_type": type(e).__name__,
                        "error_message": str(e),
                    }
                    failed_jobs += 1
                    print(f"  FAILED: {type(e).__name__}: {e}")

                writer.writerow(summary_row)
                f.flush()

    print()
    print("Batch complete.")
    print(f"Completed simulations: {completed_jobs}")
    print(f"Failed simulations: {failed_jobs}")
    print(f"Summary written to: {summary_path}")
    print(f"Trajectories written to: {trajectories_dir}")

    return summary_path
