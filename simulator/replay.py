\
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from simulator.config_loader import load_strategy_from_yaml
from simulator.execution import execute_action
from simulator.liquidity import liquidity_limits_for_action
from simulator.models import MarketTick, DecisionState, SimulationResult
from simulator.portfolio import Portfolio
from simulator.strategies import BaseStrategy, StrategyDecision


BASE_MARKET_COLUMNS = {
    "timestamp",
    "unix_time",
    "seconds_left",
    "elapsed",
    "up_price",
    "down_price",
    "btc_binance",
    "btc_chainlink",
    "price_to_beat",
}


def parse_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def load_market_ticks(csv_path: str | Path) -> list[MarketTick]:
    csv_path = Path(csv_path)
    ticks: list[MarketTick] = []

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            up_price = parse_float(row.get("up_price"))
            down_price = parse_float(row.get("down_price"))

            # Skip rows that cannot be replayed as price states.
            if up_price is None or down_price is None:
                continue

            ticks.append(MarketTick(
                timestamp=row.get("timestamp", ""),
                unix_time=parse_float(row.get("unix_time")) or 0.0,
                seconds_left=parse_float(row.get("seconds_left")),
                elapsed=parse_float(row.get("elapsed")),
                up_price=up_price,
                down_price=down_price,
                btc_binance=parse_float(row.get("btc_binance")),
                btc_chainlink=parse_float(row.get("btc_chainlink")),
                price_to_beat=parse_float(row.get("price_to_beat")),
                extra={
                    key: value
                    for key, value in row.items()
                    if key not in BASE_MARKET_COLUMNS
                },
            ))

    return ticks


def infer_final_outcome(ticks: Iterable[MarketTick], fallback: str | None = None) -> str:
    """Infer final outcome from last BTC and price_to_beat.

    This is only a fallback. Later, it is better to read actual resolution rows
    or Polymarket market outcome data.
    """
    last_btc = None
    last_ptb = None

    for tick in ticks:
        btc = tick.btc_chainlink if tick.btc_chainlink is not None else tick.btc_binance
        if btc is not None:
            last_btc = btc
        if tick.price_to_beat is not None:
            last_ptb = tick.price_to_beat

    if last_btc is not None and last_ptb is not None:
        return "up" if last_btc >= last_ptb else "down"

    if fallback is not None:
        return fallback.lower().strip()

    raise ValueError("Could not infer final outcome. Pass --final-outcome up/down.")


def run_simulation(
    *,
    market_csv: str | Path,
    strategy: BaseStrategy,
    output_csv: str | Path,
    starting_balance: float = 100.0,
    order_usd: float = 1.0,
    final_outcome: str | None = None,
    liquidity_aware_execution: bool = False,
    liquidity_depth_window_cents: int = 2,
    liquidity_fill_fraction: float = 1.0,
    liquidity_missing_depth_policy: str = "skip",
) -> SimulationResult:
    market_csv = Path(market_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    ticks = load_market_ticks(market_csv)
    if not ticks:
        raise ValueError(f"No replayable ticks found in {market_csv}")

    resolved_outcome = infer_final_outcome(ticks, fallback=final_outcome)
    portfolio = Portfolio(cash=starting_balance)
    market_start_balance = starting_balance

    rows = []
    last_action = "none"
    orders_placed = 0
    market_spend_used = 0.0

    for tick in ticks:
        current_balance = portfolio.mark_to_market(tick.up_price, tick.down_price)

        state = DecisionState(
            tick=tick,
            cash=portfolio.cash,
            up_tokens=portfolio.up_tokens,
            down_tokens=portfolio.down_tokens,
            current_balance=current_balance,
            market_start_balance=market_start_balance,
            market_spend_used=market_spend_used,
            last_action=last_action,
            orders_placed=orders_placed,
        )

        if tick.seconds_left is not None and tick.seconds_left <= 0:
            decision = StrategyDecision("hold", "market closed")
        else:
            decision = strategy.decide(state)
        decision_usd_amount = decision.usd_amount if decision.usd_amount is not None else order_usd
        liquidity = {
            "requested_usd_amount": decision_usd_amount,
            "executable_usd_amount": decision_usd_amount,
            "max_buy_usd": "",
            "max_sell_tokens": "",
            "reason": "liquidity disabled",
        }
        execution_usd_amount = decision_usd_amount
        max_buy_usd = None
        max_sell_tokens = None
        if liquidity_aware_execution:
            liquidity = liquidity_limits_for_action(
                action=decision.action,
                row_metrics=tick.extra,
                requested_usd=decision_usd_amount,
                cash=portfolio.cash,
                up_tokens=portfolio.up_tokens,
                down_tokens=portfolio.down_tokens,
                depth_window_cents=liquidity_depth_window_cents,
                fill_fraction=liquidity_fill_fraction,
                missing_depth_policy=liquidity_missing_depth_policy,
            )
            execution_usd_amount = liquidity["executable_usd_amount"]
            max_buy_usd = liquidity["max_buy_usd"]
            max_sell_tokens = liquidity["max_sell_tokens"]
        events = execute_action(
            portfolio=portfolio,
            action=decision.action,
            timestamp=tick.timestamp,
            up_price=tick.up_price,
            down_price=tick.down_price,
            usd_amount=execution_usd_amount,
            max_buy_usd=max_buy_usd,
            max_sell_tokens=max_sell_tokens,
            reason=decision.reason,
        )

        if events:
            orders_placed += len(events)
            market_spend_used += sum(event.usd_amount for event in events if event.action == "buy")

        balance_after = portfolio.mark_to_market(tick.up_price, tick.down_price)

        replay_row = {
            "timestamp": tick.timestamp,
            "unix_time": tick.unix_time,
            "seconds_left": tick.seconds_left,
            "elapsed": tick.elapsed,
            "up_price": tick.up_price,
            "down_price": tick.down_price,
            "btc_binance": tick.btc_binance,
            "btc_chainlink": tick.btc_chainlink,
            "price_to_beat": tick.price_to_beat,
            "cash_before": state.cash,
            "up_tokens_before": state.up_tokens,
            "down_tokens_before": state.down_tokens,
            "balance_before": state.current_balance,
            "action": decision.action,
            "reason": decision.reason,
            "usd_amount": decision_usd_amount,
            "executed_usd_amount": execution_usd_amount,
            "liquidity_aware_execution": liquidity_aware_execution,
            "liquidity_depth_window_cents": liquidity_depth_window_cents if liquidity_aware_execution else "",
            "liquidity_fill_fraction": liquidity_fill_fraction if liquidity_aware_execution else "",
            "liquidity_requested_usd_amount": liquidity["requested_usd_amount"],
            "liquidity_executable_usd_amount": liquidity["executable_usd_amount"],
            "liquidity_max_buy_usd": liquidity["max_buy_usd"],
            "liquidity_max_sell_tokens": liquidity["max_sell_tokens"],
            "liquidity_reason": liquidity["reason"],
            "events_count": len(events),
            "market_spend_used_before": market_spend_used - sum(event.usd_amount for event in events if event.action == "buy"),
            "market_spend_used_after": market_spend_used,
            "cash_after": portfolio.cash,
            "up_tokens_after": portfolio.up_tokens,
            "down_tokens_after": portfolio.down_tokens,
            "balance_after": balance_after,
        }
        replay_row.update(tick.extra)
        rows.append(replay_row)

        last_action = decision.action

    final_balance = portfolio.resolve(resolved_outcome)
    total_reward = final_balance - starting_balance

    fieldnames = list(rows[0].keys()) + [
        "final_outcome",
        "final_balance",
        "total_reward",
        "reward_to_go",
    ]

    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            row = dict(row)
            row["final_outcome"] = resolved_outcome
            row["final_balance"] = final_balance
            row["total_reward"] = total_reward
            row["reward_to_go"] = final_balance - float(row["balance_before"])
            writer.writerow(row)

    return SimulationResult(
        market_file=str(market_csv),
        strategy_name=strategy.name,
        final_outcome=resolved_outcome,
        final_balance=final_balance,
        starting_balance=starting_balance,
        total_reward=total_reward,
        rows_written=len(rows),
    )


def run_from_config(
    *,
    market_csv: str | Path,
    strategy_config: str | Path,
    output_csv: str | Path,
    final_outcome: str | None = None,
    liquidity_aware_execution: bool = False,
    liquidity_depth_window_cents: int = 2,
    liquidity_fill_fraction: float = 1.0,
    liquidity_missing_depth_policy: str = "skip",
):
    strategy, cfg = load_strategy_from_yaml(strategy_config)

    starting_balance = float(cfg.get("starting_balance", 100.0))
    order_usd = float(cfg.get("order_usd", 1.0))

    return run_simulation(
        market_csv=market_csv,
        strategy=strategy,
        output_csv=output_csv,
        starting_balance=starting_balance,
        order_usd=order_usd,
        final_outcome=final_outcome,
        liquidity_aware_execution=liquidity_aware_execution,
        liquidity_depth_window_cents=liquidity_depth_window_cents,
        liquidity_fill_fraction=liquidity_fill_fraction,
        liquidity_missing_depth_policy=liquidity_missing_depth_policy,
    )
