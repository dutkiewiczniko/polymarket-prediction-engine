import argparse
import csv
import html
import math
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.batch import (
    apply_drawdown_reserve_top_up,
    apply_reserve_top_up,
    discover_market_csvs,
    resolve_effective_market_balance,
)
from simulator.config_loader import build_strategy_from_config, load_yaml
from simulator.replay import run_simulation


def parse_args():
    parser = argparse.ArgumentParser(description="Run one strategy across independent groups of markets.")
    parser.add_argument("--markets-folder", default="simulator_ready_markets_liquidity_all_20260519")
    parser.add_argument("--market-pattern", default="btc-updown-5m-*.csv")
    parser.add_argument(
        "--strategy-config",
        default="configs/strategies/overnight_live_15/up_bias_chaser_tp095_cheap0025_tok20.yaml",
    )
    parser.add_argument(
        "--balance-config",
        default="configs/live_real_start110_reserve50_balance_bands.yaml",
    )
    parser.add_argument("--groups", type=int, default=50)
    parser.add_argument("--markets-per-group", type=int, default=6)
    parser.add_argument(
        "--market-offset",
        type=int,
        default=0,
        help="Skip this many sequential markets (sorted by filename) before taking groups x markets-per-group.",
    )
    parser.add_argument("--batch-id", default="")
    parser.add_argument("--output-root", default="runs")
    parser.add_argument("--min-order-usd", type=float, default=1.0)
    parser.add_argument("--liquidity-depth-window-cents", type=int, default=2)
    parser.add_argument("--liquidity-fill-fraction", type=float, default=1.0)
    parser.add_argument(
        "--liquidity-missing-depth-policy",
        choices=["skip", "allow"],
        default="skip",
    )
    parser.add_argument(
        "--trajectory-log-mode",
        choices=["none", "actions", "all"],
        default="actions",
        help="Store no trajectories, action/event rows only, or all replay rows.",
    )
    return parser.parse_args()


def safe_name(value: str) -> str:
    value = str(value).strip()
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value)
    return value.strip("_") or "unnamed"


MARKET_INTERVAL_S = 300


def find_prior_market_csv(market_path: Path) -> Path | None:
    """Markets are named btc-updown-5m-<unix_start>.csv; the previous market is
    <unix_start - 300> in the same folder. Returns None when no such file exists
    (gap in the recorded series)."""
    stem_ts = market_path.stem.rsplit("-", 1)[-1]
    if not stem_ts.isdigit():
        return None
    prior = market_path.with_name(f"{market_path.stem[:-len(stem_ts)]}{int(stem_ts) - MARKET_INTERVAL_S}.csv")
    return prior if prior.exists() else None


def find_prior_market_chain(market_path: Path, depth: int) -> list[Path]:
    """Walk back up to `depth` consecutive prior markets, oldest first, stopping
    at the first gap. A contiguous chain is required -- seeding across a gap
    would splice stale prices into the momentum series."""
    chain: list[Path] = []
    current = market_path
    for _ in range(depth):
        prior = find_prior_market_csv(current)
        if prior is None:
            break
        chain.append(prior)
        current = prior
    chain.reverse()
    return chain


def load_btc_warmup_samples(market_csv: Path) -> list[tuple[float, float]]:
    """Extract the (unix_time, btc_price) series from a market CSV for seeding
    strategy momentum history. Chainlink preferred, Binance fallback, matching
    the strategy engine's own choice."""
    samples = []
    with market_csv.open("r", newline="", encoding="utf-8-sig") as f:
        for row in csv.DictReader(f):
            try:
                unix_time = float(row["unix_time"])
            except (KeyError, TypeError, ValueError):
                continue
            btc = row.get("btc_chainlink") or row.get("btc_binance")
            try:
                samples.append((unix_time, float(btc)))
            except (TypeError, ValueError):
                continue
    return samples


def select_market_group(
    markets_folder: str,
    market_pattern: str,
    groups: int,
    markets_per_group: int,
    offset: int = 0,
) -> list[Path]:
    """Pick `groups * markets_per_group` sequential markets (sorted by filename),
    starting `offset` markets into the folder. Same offset always yields the same
    markets, which is what a fair before/after comparison needs; vary `offset` to
    re-run a check against a different historical slice.
    """
    markets = discover_market_csvs(markets_folder, market_pattern)
    needed = groups * markets_per_group
    if offset < 0 or offset + needed > len(markets):
        raise SystemExit(
            f"Need {needed} markets starting at offset {offset}, "
            f"but only {len(markets)} markets exist in {markets_folder}"
        )
    return markets[offset:offset + needed]


def apply_virtual_withdrawals(master_balance: float, reserve: float, triggered: set[float], cfg: dict) -> tuple[float, float, float]:
    thresholds = cfg.get("withdrawal_thresholds") or []
    pct = float(cfg.get("withdrawal_pct", 0.0) or 0.0)
    if not thresholds or pct <= 0:
        return master_balance, reserve, 0.0

    withdrawn = 0.0
    for threshold in sorted(float(value) for value in thresholds):
        if threshold in triggered:
            continue
        if master_balance >= threshold:
            amount = master_balance * pct
            master_balance = max(0.0, master_balance - amount)
            reserve += amount
            triggered.add(threshold)
            withdrawn += amount
    return master_balance, reserve, withdrawn


def count_events(df: pd.DataFrame, action: str, side: str) -> int:
    if "events_count" not in df.columns:
        return 0
    mask = df["events_count"].fillna(0).astype(float).gt(0)
    if "action" in df.columns:
        mask &= df["action"].astype(str).eq(action)
    if side == "up" and "up_tokens_after" in df.columns and "up_tokens_before" in df.columns:
        mask &= pd.to_numeric(df["up_tokens_after"], errors="coerce").fillna(0).gt(
            pd.to_numeric(df["up_tokens_before"], errors="coerce").fillna(0)
        )
    if side == "down" and "down_tokens_after" in df.columns and "down_tokens_before" in df.columns:
        mask &= pd.to_numeric(df["down_tokens_after"], errors="coerce").fillna(0).gt(
            pd.to_numeric(df["down_tokens_before"], errors="coerce").fillna(0)
        )
    return int(mask.sum())


def summarize_trajectory(path: Path, mode: str) -> dict:
    if not path.exists():
        return {
            "orders_placed": 0,
            "capped_rows": 0,
            "requested_usd": 0.0,
            "executed_usd": 0.0,
            "max_order_usd": 0.0,
            "buy_up_events": 0,
            "buy_down_events": 0,
            "sell_rows": 0,
            "rows_written": 0,
        }

    df = pd.read_csv(path)
    events = pd.to_numeric(df.get("events_count", 0), errors="coerce").fillna(0).gt(0)
    event_df = df[events].copy()
    capped = df.get("liquidity_reason", pd.Series(dtype=str)).fillna("").astype(str).str.contains("capped", case=False)
    requested = pd.to_numeric(event_df.get("usd_amount", 0), errors="coerce").fillna(0)
    executed = pd.to_numeric(event_df.get("executed_usd_amount", 0), errors="coerce").fillna(0)

    result = {
        "orders_placed": int(events.sum()),
        "capped_rows": int(capped.sum()),
        "requested_usd": float(requested.sum()),
        "executed_usd": float(executed.sum()),
        "max_order_usd": float(executed.max()) if not executed.empty else 0.0,
        "buy_up_events": count_events(df, "buy_up", "up"),
        "buy_down_events": count_events(df, "buy_down", "down"),
        "sell_rows": int(event_df["action"].fillna("").astype(str).str.startswith("sell_").sum()) if "action" in event_df else 0,
        "rows_written": len(df),
    }

    if mode == "actions":
        keep = df[
            df.get("action", pd.Series([""] * len(df))).fillna("").astype(str).ne("hold")
            | events
        ].copy()
        keep.to_csv(path, index=False)
        result["rows_written"] = len(keep)
    elif mode == "none":
        path.unlink(missing_ok=True)
        result["rows_written"] = 0

    return result


def fmt_money(value) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return "--"


def fmt_pct(value) -> str:
    try:
        return f"{100.0 * float(value):.1f}%"
    except (TypeError, ValueError):
        return "--"


def sparkline(values, *, width=220, height=54, color="#2563eb") -> str:
    values = [float(v) for v in values if v is not None and not math.isnan(float(v))]
    if not values:
        return f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}"></svg>'
    if len(values) == 1:
        values = [values[0], values[0]]
    lo = min(values)
    hi = max(values)
    pad = 4
    span = hi - lo if hi != lo else 1.0
    step = (width - pad * 2) / (len(values) - 1)
    points = []
    for i, value in enumerate(values):
        x = pad + i * step
        y = height - pad - ((value - lo) / span) * (height - pad * 2)
        points.append(f"{x:.1f},{y:.1f}")
    zero_line = ""
    if lo < 0 < hi:
        zy = height - pad - ((0 - lo) / span) * (height - pad * 2)
        zero_line = f'<line x1="{pad}" y1="{zy:.1f}" x2="{width-pad}" y2="{zy:.1f}" stroke="#d1d5db" stroke-width="1"/>'
    return (
        f'<svg width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
        f'{zero_line}<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(points)}"/>'
        f'<circle cx="{points[-1].split(",")[0]}" cy="{points[-1].split(",")[1]}" r="2.5" fill="{color}"/>'
        "</svg>"
    )


def write_report(run_dir: Path, group_df: pd.DataFrame, market_df: pd.DataFrame, args):
    initial_master = float(group_df["start_master"].iloc[0]) if not group_df.empty else 0.0
    initial_reserve = float(group_df["start_reserve"].iloc[0]) if not group_df.empty else 0.0
    initial_total = initial_master + initial_reserve
    group_df = group_df.copy()
    group_df["final_total_equity"] = group_df["final_master"] + group_df["final_reserve"]
    group_df["total_equity_reward"] = group_df["final_total_equity"] - initial_total
    market_df = market_df.copy()

    mean_final_master = group_df["final_master"].mean()
    median_final_master = group_df["final_master"].median()
    mean_total_equity = group_df["final_total_equity"].mean()
    mean_reward = group_df["group_reward"].mean()
    win_rate = (group_df["group_reward"] > 0).mean()
    loss_rate = (group_df["group_reward"] < 0).mean()
    bust_rate = (group_df["final_master"] < 1).mean()
    mean_orders = group_df["orders_placed"].mean()
    cap_rate = (market_df["capped_rows"] > 0).mean() if not market_df.empty else 0.0
    clipped_usd = (market_df["requested_usd"] - market_df["executed_usd"]).sum()

    by_position = market_df.groupby("market_in_group").agg(
        mean_master_after=("master_after", "mean"),
        median_master_after=("master_after", "median"),
        mean_reward=("reward", "mean"),
        win_rate=("reward", lambda s: float((s > 0).mean())),
        mean_orders=("orders_placed", "mean"),
    ).reset_index()
    avg_path = [initial_master] + by_position["mean_master_after"].tolist()

    cards = [
        ("Groups", f"{len(group_df)}"),
        ("Markets", f"{len(market_df)}"),
        ("Mean Final Master", fmt_money(mean_final_master)),
        ("Median Final Master", fmt_money(median_final_master)),
        ("Mean Total Equity", fmt_money(mean_total_equity)),
        ("Mean Group PnL", fmt_money(mean_reward)),
        ("Group Win Rate", fmt_pct(win_rate)),
        ("Bust Rate", fmt_pct(bust_rate)),
        ("Mean Orders / Group", f"{mean_orders:.1f}"),
        ("Markets Capped", fmt_pct(cap_rate)),
        ("Liquidity Clipped", fmt_money(clipped_usd)),
    ]
    card_html = "\n".join(
        f'<div class="card"><div class="label">{html.escape(label)}</div><div class="value">{html.escape(value)}</div></div>'
        for label, value in cards
    )

    band_rows = "\n".join(
        f"<tr><td>{int(row.market_in_group)}</td><td>{fmt_money(row.mean_master_after)}</td>"
        f"<td>{fmt_money(row.median_master_after)}</td><td>{fmt_money(row.mean_reward)}</td>"
        f"<td>{fmt_pct(row.win_rate)}</td><td>{row.mean_orders:.1f}</td></tr>"
        for row in by_position.itertuples(index=False)
    )

    compact_groups = "\n".join(
        f'<div class="mini {"win" if row.group_reward > 0 else "loss" if row.group_reward < 0 else ""}">'
        f'<div class="mini-head"><b>G{int(row.group):02d}</b><span>M {fmt_money(row.final_master)} | R {fmt_money(row.final_reserve)}</span></div>'
        f'{sparkline([float(x) for x in str(row.balance_path).split("|")], width=170, height=44)}'
        f'<div class="mini-sub">PnL {fmt_money(row.group_reward)} | equity {fmt_money(row.final_total_equity)} | orders {int(row.orders_placed)}</div>'
        "</div>"
        for row in group_df.itertuples(index=False)
    )

    group_rows = "\n".join(
        f"<tr><td>{int(row.group)}</td><td>{fmt_money(row.start_master)}</td><td>{fmt_money(row.final_master)}</td>"
        f"<td>{fmt_money(row.final_reserve)}</td><td>{fmt_money(row.group_reward)}</td>"
        f"<td>{int(row.winning_markets)}</td><td>{int(row.losing_markets)}</td>"
        f"<td>{int(row.orders_placed)}</td><td>{int(row.capped_rows)}</td><td>{fmt_money(row.max_order_usd)}</td></tr>"
        for row in group_df.itertuples(index=False)
    )

    market_rows = "\n".join(
        f"<tr><td>{int(row.group)}</td><td>{int(row.market_in_group)}</td><td>{html.escape(Path(row.market_file).name)}</td>"
        f"<td>{html.escape(str(row.final_outcome))}</td><td>{fmt_money(row.effective_balance)}</td>"
        f"<td>{fmt_money(row.reward)}</td><td>{fmt_money(row.master_after)}</td><td>{fmt_money(row.reserve_after)}</td><td>{fmt_money(row.total_equity_after)}</td>"
        f"<td>{int(row.orders_placed)}</td><td>{int(row.capped_rows)}</td>"
        f"<td>{fmt_money(row.executed_usd)}</td><td>{fmt_money(row.max_order_usd)}</td></tr>"
        for row in market_df.itertuples(index=False)
    )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Grouped Strategy Report</title>
  <style>
    body {{ margin:0; font-family:Arial, sans-serif; background:#f6f8fb; color:#172033; }}
    header {{ padding:22px 28px 14px; background:#ffffff; border-bottom:1px solid #d8e0ec; }}
    h1 {{ margin:0 0 6px; font-size:24px; }}
    h2 {{ margin:24px 0 10px; font-size:17px; }}
    main {{ padding:18px 28px 34px; }}
    .muted {{ color:#64748b; font-size:13px; }}
    .cards {{ display:grid; grid-template-columns:repeat(6, minmax(130px, 1fr)); gap:10px; }}
    .card {{ background:#fff; border:1px solid #d8e0ec; border-radius:8px; padding:12px; }}
    .label {{ color:#64748b; font-size:12px; }}
    .value {{ font-weight:700; font-size:20px; margin-top:4px; }}
    .panel {{ background:#fff; border:1px solid #d8e0ec; border-radius:8px; padding:14px; margin-top:12px; }}
    .top-grid {{ display:grid; grid-template-columns:1fr 1fr; gap:14px; align-items:start; }}
    table {{ width:100%; border-collapse:collapse; background:#fff; }}
    th,td {{ padding:7px 8px; border-bottom:1px solid #e5ebf3; text-align:left; font-size:12px; }}
    th {{ color:#315f9f; background:#f8fbff; position:sticky; top:0; }}
    .mini-grid {{ display:grid; grid-template-columns:repeat(5, minmax(160px, 1fr)); gap:10px; }}
    .mini {{ background:#fff; border:1px solid #d8e0ec; border-radius:8px; padding:8px; }}
    .mini.win {{ border-left:4px solid #16a34a; }}
    .mini.loss {{ border-left:4px solid #dc2626; }}
    .mini-head {{ display:flex; justify-content:space-between; font-size:13px; }}
    .mini-sub {{ color:#64748b; font-size:11px; }}
    .scroll {{ max-height:440px; overflow:auto; border:1px solid #d8e0ec; border-radius:8px; }}
    .avg-path {{ display:flex; align-items:center; gap:16px; }}
    @media (max-width:1100px) {{
      .cards {{ grid-template-columns:repeat(3, 1fr); }}
      .top-grid {{ grid-template-columns:1fr; }}
      .mini-grid {{ grid-template-columns:repeat(2, 1fr); }}
    }}
  </style>
</head>
<body>
<header>
  <h1>Grouped Up-Bias Chaser Backtest</h1>
  <div class="muted">
    Strategy: {html.escape(args.strategy_config)} | Balance config: {html.escape(args.balance_config)} |
    {args.groups} groups x {args.markets_per_group} markets | liquidity window {args.liquidity_depth_window_cents}c,
    fill fraction {args.liquidity_fill_fraction}
  </div>
</header>
<main>
  <section class="cards">{card_html}</section>

  <section class="top-grid">
    <div class="panel">
      <h2>Average Tradable Master Path</h2>
      <div class="avg-path">{sparkline(avg_path, width=460, height=130)}<div>
        <div>Start master: <b>{fmt_money(initial_master)}</b></div>
        <div>Initial reserve: <b>{fmt_money(initial_reserve)}</b></div>
        <div>Mean final master: <b>{fmt_money(mean_final_master)}</b></div>
      </div></div>
    </div>
    <div class="panel">
      <h2>Averages By Market Position In Group</h2>
      <table><thead><tr><th>Market #</th><th>Mean Master After</th><th>Median Master After</th><th>Mean Reward</th><th>Win Rate</th><th>Mean Orders</th></tr></thead><tbody>{band_rows}</tbody></table>
    </div>
  </section>

  <section>
    <h2>All 50 Group Balance Paths</h2>
    <div class="mini-grid">{compact_groups}</div>
  </section>

  <section class="panel">
    <h2>Group Summary</h2>
    <table><thead><tr><th>Group</th><th>Start Master</th><th>Final Master</th><th>Final Reserve</th><th>PnL</th><th>Wins</th><th>Losses</th><th>Orders</th><th>Capped Rows</th><th>Max Order</th></tr></thead><tbody>{group_rows}</tbody></table>
  </section>

  <section>
    <h2>Market Details</h2>
    <div class="scroll"><table><thead><tr><th>Group</th><th>#</th><th>Market</th><th>Outcome</th><th>Eff Bal</th><th>Reward</th><th>Master After</th><th>Reserve After</th><th>Total Equity</th><th>Orders</th><th>Capped</th><th>Executed USD</th><th>Max Order</th></tr></thead><tbody>{market_rows}</tbody></table></div>
  </section>
</main>
</body>
</html>"""
    (run_dir / "grouped_strategy_report.html").write_text(html_text, encoding="utf-8")


def run_grouped_strategy(
    markets: list[Path],
    strategy_cfg: dict,
    balance_cfg: dict,
    *,
    groups: int,
    markets_per_group: int,
    trajectories_dir: Path,
    min_order_usd: float = 1.0,
    liquidity_depth_window_cents: int = 2,
    liquidity_fill_fraction: float = 1.0,
    liquidity_missing_depth_policy: str = "skip",
    trajectory_log_mode: str = "actions",
    warmup_prior_market: bool = False,
    verbose: bool = True,
) -> tuple["pd.DataFrame", "pd.DataFrame"]:
    """Simulate one strategy across `groups` sequential, non-overlapping chunks of
    `markets_per_group` markets each, carrying a running master/reserve balance
    within each group. Returns (market_df, group_df). Caller is responsible for
    picking `markets` (see select_market_group) so the same markets can be reused
    across strategies for a fair comparison.
    """
    needed = groups * markets_per_group
    if len(markets) < needed:
        raise SystemExit(f"Need {needed} markets, got {len(markets)}")

    start_master = float(balance_cfg.get("starting_balance", 100.0))
    initial_reserve = float(balance_cfg.get("initial_reserved_balance", 0.0) or 0.0)

    market_rows = []
    group_rows = []

    for group_index in range(groups):
        group_markets = markets[
            group_index * markets_per_group:(group_index + 1) * markets_per_group
        ]
        master = start_master
        reserve = initial_reserve
        triggered = set()
        balance_path = [master]
        reserve_path = [reserve]
        master_history = [master]
        cooldown_markets_remaining = 0
        group_market_rewards = []
        group_orders = 0
        group_capped = 0
        group_requested = 0.0
        group_executed = 0.0
        group_max_order = 0.0

        if verbose:
            print(f"Group {group_index + 1}/{groups}")
        for market_in_group, market_path in enumerate(group_markets, start=1):
            master, reserve, topped_up = apply_reserve_top_up(master, reserve, balance_cfg)
            master_before = master
            if cooldown_markets_remaining > 0:
                cooldown_markets_remaining -= 1
                reward = 0.0
                balance_path.append(master)
                master_history.append(master)
                reserve_path.append(reserve)
                group_market_rewards.append(reward)
                market_rows.append({
                    "group": group_index + 1,
                    "market_in_group": market_in_group,
                    "market_file": str(market_path),
                    "final_outcome": "skipped",
                    "reserve_top_up_before_market": topped_up,
                    "drawdown_top_up_after_market": 0.0,
                    "cooldown_markets_remaining": cooldown_markets_remaining,
                    "master_before": master_before,
                    "effective_balance": 0.0,
                    "market_final_balance": 0.0,
                    "reward": reward,
                    "master_after_before_withdrawal": master,
                    "withdrawn_after_market": 0.0,
                    "master_after": master,
                    "reserve_after": reserve,
                    "total_equity_after": master + reserve,
                    "orders_placed": 0,
                    "buy_up_events": 0,
                    "buy_down_events": 0,
                    "sell_rows": 0,
                    "capped_rows": 0,
                    "requested_usd": 0.0,
                    "executed_usd": 0.0,
                    "max_order_usd": 0.0,
                    "rows_written": 0,
                    "output_csv": "",
                })
                continue

            effective_balance = resolve_effective_market_balance(master_before, balance_cfg)
            if effective_balance is None:
                effective_balance = master_before
            effective_balance = float(effective_balance)
            output_csv = trajectories_dir / f"group_{group_index + 1:03d}" / f"{market_path.stem}.csv"
            strategy = build_strategy_from_config(strategy_cfg)
            if warmup_prior_market:
                from simulator.strategies import RuleBasedStrategy
                depth = -(-int(max(RuleBasedStrategy.MOMENTUM_WINDOWS_S)) // MARKET_INTERVAL_S)
                for prior_csv in find_prior_market_chain(market_path, depth):
                    strategy.seed_btc_history(load_btc_warmup_samples(prior_csv))
            result = run_simulation(
                market_csv=market_path,
                strategy=strategy,
                output_csv=output_csv,
                starting_balance=effective_balance,
                order_usd=float(strategy_cfg.get("order_usd", 1.0)),
                final_outcome=None,
                liquidity_aware_execution=True,
                liquidity_depth_window_cents=liquidity_depth_window_cents,
                liquidity_fill_fraction=liquidity_fill_fraction,
                liquidity_missing_depth_policy=liquidity_missing_depth_policy,
                min_order_usd=min_order_usd,
            )
            trajectory_stats = summarize_trajectory(output_csv, trajectory_log_mode)
            reward = float(result.total_reward)
            master_after_before_withdrawal = max(0.0, master_before + reward)
            master, reserve, withdrawn = apply_virtual_withdrawals(
                master_after_before_withdrawal,
                reserve,
                triggered,
                balance_cfg,
            )
            master_history.append(master)
            master, reserve, drawdown_topped_up, drawdown_triggered = apply_drawdown_reserve_top_up(
                master,
                reserve,
                master_history,
                balance_cfg,
            )
            if drawdown_triggered:
                master_history = [master]
                cooldown_markets_remaining = max(
                    cooldown_markets_remaining,
                    int(balance_cfg.get("reserve_drawdown_skip_markets", 0) or 0),
                )
            balance_path.append(master)
            reserve_path.append(reserve)
            group_market_rewards.append(reward)
            group_orders += trajectory_stats["orders_placed"]
            group_capped += trajectory_stats["capped_rows"]
            group_requested += trajectory_stats["requested_usd"]
            group_executed += trajectory_stats["executed_usd"]
            group_max_order = max(group_max_order, trajectory_stats["max_order_usd"])

            market_rows.append({
                "group": group_index + 1,
                "market_in_group": market_in_group,
                "market_file": str(market_path),
                "final_outcome": result.final_outcome,
                "reserve_top_up_before_market": topped_up,
                "drawdown_top_up_after_market": drawdown_topped_up,
                "cooldown_markets_remaining": cooldown_markets_remaining,
                "master_before": master_before,
                "effective_balance": effective_balance,
                "market_final_balance": result.final_balance,
                "reward": reward,
                "master_after_before_withdrawal": master_after_before_withdrawal,
                "withdrawn_after_market": withdrawn,
                "master_after": master,
                "reserve_after": reserve,
                "total_equity_after": master + reserve,
                "orders_placed": trajectory_stats["orders_placed"],
                "buy_up_events": trajectory_stats["buy_up_events"],
                "buy_down_events": trajectory_stats["buy_down_events"],
                "sell_rows": trajectory_stats["sell_rows"],
                "capped_rows": trajectory_stats["capped_rows"],
                "requested_usd": trajectory_stats["requested_usd"],
                "executed_usd": trajectory_stats["executed_usd"],
                "max_order_usd": trajectory_stats["max_order_usd"],
                "rows_written": trajectory_stats["rows_written"],
                "output_csv": str(output_csv if output_csv.exists() else ""),
            })

        group_rows.append({
            "group": group_index + 1,
            "markets": len(group_markets),
            "start_master": start_master,
            "start_reserve": initial_reserve,
            "final_master": master,
            "final_reserve": reserve,
            "final_total_equity": master + reserve,
            "group_reward": master - start_master,
            "total_equity_reward": master + reserve - start_master - initial_reserve,
            "winning_markets": sum(1 for value in group_market_rewards if value > 0),
            "losing_markets": sum(1 for value in group_market_rewards if value < 0),
            "flat_markets": sum(1 for value in group_market_rewards if value == 0),
            "orders_placed": group_orders,
            "capped_rows": group_capped,
            "requested_usd": group_requested,
            "executed_usd": group_executed,
            "max_order_usd": group_max_order,
            "balance_path": "|".join(f"{value:.10g}" for value in balance_path),
            "reserve_path": "|".join(f"{value:.10g}" for value in reserve_path),
        })

    return pd.DataFrame(market_rows), pd.DataFrame(group_rows)


def main():
    args = parse_args()
    strategy_cfg = load_yaml(args.strategy_config)
    balance_cfg = load_yaml(args.balance_config)
    strategy_name = strategy_cfg.get("name") or Path(args.strategy_config).stem
    batch_id = args.batch_id or f"grouped_{safe_name(strategy_name)}_{args.groups}x{args.markets_per_group}"
    run_dir = Path(args.output_root) / safe_name(batch_id)
    trajectories_dir = run_dir / "trajectories"
    trajectories_dir.mkdir(parents=True, exist_ok=True)

    markets = select_market_group(
        args.markets_folder, args.market_pattern, args.groups, args.markets_per_group, args.market_offset
    )
    needed = args.groups * args.markets_per_group

    print(f"Grouped backtest: {batch_id}")
    print(f"Markets: {needed} ({args.groups} groups x {args.markets_per_group}, offset {args.market_offset})")
    print(f"Run dir: {run_dir}")

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

    run_dir.mkdir(parents=True, exist_ok=True)
    market_df.to_csv(run_dir / "market_summary.csv", index=False)
    group_df.to_csv(run_dir / "group_summary.csv", index=False)

    with (run_dir / "run_config.txt").open("w", encoding="utf-8") as f:
        for key, value in sorted(vars(args).items()):
            f.write(f"{key}: {value}\n")

    write_report(run_dir, group_df, market_df, args)
    print(f"Group summary: {run_dir / 'group_summary.csv'}")
    print(f"Market summary: {run_dir / 'market_summary.csv'}")
    print(f"Report: {run_dir / 'grouped_strategy_report.html'}")


if __name__ == "__main__":
    main()
