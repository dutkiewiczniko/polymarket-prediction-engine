import argparse
import html
import random
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.batch import discover_market_csvs
from simulator.config_loader import load_yaml
from scripts.simulation.run_grouped_backtest import (
    run_grouped_strategy,
    safe_name,
    select_market_group,
    write_report,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "A/B regression harness: run two or more strategy configs across the same "
            "sequential groups of markets so results are directly comparable."
        )
    )
    parser.add_argument("--strategy-configs", nargs="+", required=True, help="2+ strategy config YAML paths.")
    parser.add_argument(
        "--labels",
        nargs="*",
        default=None,
        help="Optional label per strategy config (defaults to each config's stem).",
    )
    parser.add_argument("--markets-folder", default="simulator_ready_markets_liquidity_all_20260519")
    parser.add_argument("--market-pattern", default="btc-updown-5m-*.csv")
    parser.add_argument("--groups", type=int, default=5)
    parser.add_argument("--markets-per-group", type=int, default=20)
    parser.add_argument(
        "--market-offset",
        type=int,
        default=0,
        help="Skip this many sequential markets before taking groups x markets-per-group. "
        "Same offset -> same markets -> a fair before/after comparison. Vary it to re-check "
        "against a different historical slice.",
    )
    parser.add_argument(
        "--random-market-offset",
        action="store_true",
        help="Pick a random valid offset instead of --market-offset (still identical for all "
        "configs being compared in this run).",
    )
    parser.add_argument("--seed", type=int, default=None, help="Seed for --random-market-offset.")
    parser.add_argument("--balance-config", default="configs/live_real_start110_reserve50_balance_bands.yaml")
    parser.add_argument("--min-order-usd", type=float, default=1.0)
    parser.add_argument("--liquidity-depth-window-cents", type=int, default=2)
    parser.add_argument("--liquidity-fill-fraction", type=float, default=1.0)
    parser.add_argument("--liquidity-missing-depth-policy", choices=["skip", "allow"], default="skip")
    parser.add_argument("--trajectory-log-mode", choices=["none", "actions", "all"], default="none")
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--output-root", default="runs/compare")
    return parser.parse_args()


def resolve_offset(args, market_count: int) -> int:
    needed = args.groups * args.markets_per_group
    if args.random_market_offset:
        rng = random.Random(args.seed)
        max_offset = max(0, market_count - needed)
        return rng.randint(0, max_offset)
    return args.market_offset


def summarize_group_df(group_df: pd.DataFrame) -> dict:
    final_equity = group_df["final_master"] + group_df["final_reserve"]
    return {
        "groups": len(group_df),
        "mean_group_reward": group_df["group_reward"].mean(),
        "median_group_reward": group_df["group_reward"].median(),
        "mean_final_master": group_df["final_master"].mean(),
        "mean_final_equity": final_equity.mean(),
        "group_win_rate": float((group_df["group_reward"] > 0).mean()),
        "bust_rate": float((group_df["final_master"] < 1).mean()),
        "mean_orders_per_group": group_df["orders_placed"].mean(),
        "total_orders": int(group_df["orders_placed"].sum()),
    }


def fmt(value, pct=False) -> str:
    if pct:
        return f"{100.0 * float(value):.1f}%"
    return f"{float(value):,.3f}"


def write_comparison_report(run_dir: Path, comparison_df: pd.DataFrame, labels: list[str], args, market_count: int, offset: int):
    baseline_label = labels[0]
    rows = "\n".join(
        f"<tr><td>{html.escape(str(row.label))}</td><td>{fmt(row.mean_group_reward)}</td>"
        f"<td>{fmt(row.median_group_reward)}</td><td>{fmt(row.group_win_rate, pct=True)}</td>"
        f"<td>{fmt(row.bust_rate, pct=True)}</td><td>{fmt(row.mean_final_equity)}</td>"
        f"<td>{fmt(row.mean_orders_per_group)}</td>"
        f"<td>{fmt(row.mean_group_reward - comparison_df.iloc[0].mean_group_reward)}</td></tr>"
        for row in comparison_df.itertuples(index=False)
    )
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Strategy Comparison</title>
  <style>
    body {{ margin:0; font-family:Arial, sans-serif; background:#f6f8fb; color:#172033; }}
    header {{ padding:22px 28px 14px; background:#fff; border-bottom:1px solid #d8e0ec; }}
    h1 {{ margin:0 0 6px; font-size:22px; }}
    main {{ padding:18px 28px 34px; }}
    .muted {{ color:#64748b; font-size:13px; }}
    table {{ width:100%; border-collapse:collapse; background:#fff; margin-top:14px; }}
    th,td {{ padding:8px 10px; border-bottom:1px solid #e5ebf3; text-align:left; font-size:13px; }}
    th {{ color:#315f9f; background:#f8fbff; }}
    tr:first-child td {{ font-weight:700; }}
  </style>
</head>
<body>
<header>
  <h1>Strategy Comparison ({len(labels)} configs, baseline = {html.escape(baseline_label)})</h1>
  <div class="muted">
    {args.groups} groups x {args.markets_per_group} markets, offset {offset}, {market_count} markets available in
    {html.escape(args.markets_folder)} | balance config: {html.escape(args.balance_config)}
  </div>
</header>
<main>
  <table>
    <thead><tr><th>Label</th><th>Mean Group Reward</th><th>Median Group Reward</th><th>Group Win Rate</th>
    <th>Bust Rate</th><th>Mean Final Equity</th><th>Mean Orders/Group</th><th>Delta vs {html.escape(baseline_label)}</th></tr></thead>
    <tbody>{rows}</tbody>
  </table>
</main>
</body>
</html>"""
    (run_dir / "comparison_report.html").write_text(html_text, encoding="utf-8")


def main():
    args = parse_args()
    if len(args.strategy_configs) < 2:
        raise SystemExit("Provide at least 2 --strategy-configs to compare.")

    labels = args.labels or [Path(cfg).stem for cfg in args.strategy_configs]
    if len(labels) != len(args.strategy_configs):
        raise SystemExit("--labels must match --strategy-configs in count if provided.")

    balance_cfg = load_yaml(args.balance_config)
    all_markets = discover_market_csvs(args.markets_folder, args.market_pattern)
    offset = resolve_offset(args, len(all_markets))
    markets = select_market_group(
        args.markets_folder, args.market_pattern, args.groups, args.markets_per_group, offset
    )

    batch_id = args.batch_id or f"compare_{args.groups}x{args.markets_per_group}_offset{offset}"
    run_dir = Path(args.output_root) / safe_name(batch_id)
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"Comparing {len(labels)} strategy configs: {', '.join(labels)}")
    print(
        f"Markets: {len(markets)} ({args.groups} groups x {args.markets_per_group}, offset {offset}) "
        f"from {args.markets_folder} ({len(all_markets)} available) -- identical markets for every config."
    )
    print(f"Run dir: {run_dir}")

    summary_rows = []
    for label, config_path in zip(labels, args.strategy_configs):
        print(f"\n--- {label} ({config_path}) ---")
        strategy_cfg = load_yaml(config_path)
        strategy_run_dir = run_dir / safe_name(label)
        trajectories_dir = strategy_run_dir / "trajectories"
        trajectories_dir.mkdir(parents=True, exist_ok=True)

        market_df, group_df = run_grouped_strategy(
            markets,
            strategy_cfg,
            balance_cfg,
            groups=args.groups,
            markets_per_group=args.markets_per_group,
            trajectories_dir=trajectories_dir,
            min_order_usd=args.min_order_usd,
            liquidity_depth_window_cents=args.liquidity_depth_window_cents,
            liquidity_fill_fraction=args.liquidity_fill_fraction,
            liquidity_missing_depth_policy=args.liquidity_missing_depth_policy,
            trajectory_log_mode=args.trajectory_log_mode,
        )
        market_df.to_csv(strategy_run_dir / "market_summary.csv", index=False)
        group_df.to_csv(strategy_run_dir / "group_summary.csv", index=False)

        report_args = argparse.Namespace(
            strategy_config=str(config_path),
            balance_config=args.balance_config,
            groups=args.groups,
            markets_per_group=args.markets_per_group,
            liquidity_depth_window_cents=args.liquidity_depth_window_cents,
            liquidity_fill_fraction=args.liquidity_fill_fraction,
        )
        write_report(strategy_run_dir, group_df, market_df, report_args)

        stats = summarize_group_df(group_df)
        stats["label"] = label
        summary_rows.append(stats)
        print(
            f"mean group reward {stats['mean_group_reward']:.3f} | "
            f"win rate {100 * stats['group_win_rate']:.1f}% | "
            f"bust rate {100 * stats['bust_rate']:.1f}% | "
            f"orders/group {stats['mean_orders_per_group']:.1f}"
        )

    comparison_df = pd.DataFrame(summary_rows)
    comparison_df = comparison_df[
        [
            "label", "groups", "mean_group_reward", "median_group_reward", "mean_final_master",
            "mean_final_equity", "group_win_rate", "bust_rate", "mean_orders_per_group", "total_orders",
        ]
    ]
    comparison_df.to_csv(run_dir / "comparison_summary.csv", index=False)
    write_comparison_report(run_dir, comparison_df, labels, args, len(all_markets), offset)

    baseline_reward = comparison_df.iloc[0]["mean_group_reward"]
    print("\n=== Comparison summary (baseline = first config) ===")
    for row in comparison_df.itertuples(index=False):
        delta = row.mean_group_reward - baseline_reward
        print(
            f"{row.label:<40} mean_reward={row.mean_group_reward:>9.3f}  "
            f"delta={delta:>+9.3f}  win_rate={100*row.group_win_rate:>5.1f}%  "
            f"bust_rate={100*row.bust_rate:>5.1f}%"
        )
    print(f"\nComparison summary: {run_dir / 'comparison_summary.csv'}")
    print(f"Comparison report: {run_dir / 'comparison_report.html'}")


if __name__ == "__main__":
    main()
