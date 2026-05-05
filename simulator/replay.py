\
import csv
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

from simulator.config_loader import load_strategy_from_yaml
from simulator.execution import execute_action
from simulator.models import MarketTick, DecisionState, SimulationResult
from simulator.portfolio import Portfolio
from simulator.strategies import BaseStrategy


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
) -> SimulationResult:
    market_csv = Path(market_csv)
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    ticks = load_market_ticks(market_csv)
    if not ticks:
        raise ValueError(f"No replayable ticks found in {market_csv}")

    resolved_outcome = infer_final_outcome(ticks, fallback=final_outcome)
    portfolio = Portfolio(cash=starting_balance)

    rows = []
    last_action = "none"
    orders_placed = 0

    for tick in ticks:
        current_balance = portfolio.mark_to_market(tick.up_price, tick.down_price)

        state = DecisionState(
            tick=tick,
            cash=portfolio.cash,
            up_tokens=portfolio.up_tokens,
            down_tokens=portfolio.down_tokens,
            current_balance=current_balance,
            last_action=last_action,
            orders_placed=orders_placed,
        )

        decision = strategy.decide(state)
        decision_usd_amount = decision.usd_amount if decision.usd_amount is not None else order_usd
        events = execute_action(
            portfolio=portfolio,
            action=decision.action,
            timestamp=tick.timestamp,
            up_price=tick.up_price,
            down_price=tick.down_price,
            usd_amount=decision_usd_amount,
            reason=decision.reason,
        )

        if events:
            orders_placed += len(events)

        balance_after = portfolio.mark_to_market(tick.up_price, tick.down_price)

        rows.append({
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
            "events_count": len(events),
            "cash_after": portfolio.cash,
            "up_tokens_after": portfolio.up_tokens,
            "down_tokens_after": portfolio.down_tokens,
            "balance_after": balance_after,
        })

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
    )
