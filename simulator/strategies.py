\
import random
from dataclasses import dataclass
from typing import Any

from simulator.models import DecisionState


@dataclass
class StrategyDecision:
    action: str
    reason: str = ""
    usd_amount: float | None = None


class BaseStrategy:
    name = "base"

    def decide(self, state: DecisionState) -> StrategyDecision:
        return StrategyDecision("hold", "base strategy holds")


class HoldStrategy(BaseStrategy):
    name = "hold"

    def decide(self, state: DecisionState) -> StrategyDecision:
        return StrategyDecision("hold", "always hold")


class RandomStrategy(BaseStrategy):
    name = "random"

    def __init__(self, hold_probability: float = 0.70, buy_up_probability: float = 0.15, seed: int | None = None):
        self.hold_probability = hold_probability
        self.buy_up_probability = buy_up_probability
        self.rng = random.Random(seed)

    def decide(self, state: DecisionState) -> StrategyDecision:
        x = self.rng.random()
        if x < self.hold_probability:
            return StrategyDecision("hold", "random hold")
        if x < self.hold_probability + self.buy_up_probability:
            return StrategyDecision("buy_up", "random buy_up")
        return StrategyDecision("buy_down", "random buy_down")


class MomentumStrategy(BaseStrategy):
    name = "momentum_basic"

    def __init__(
        self,
        momentum_pct: float = 0.10,
        min_elapsed_s: float = 15.0,
        min_remaining_s: float = 10.0,
        min_up_prob: float = 0.30,
        max_up_prob: float = 0.70,
    ):
        self.momentum_pct = momentum_pct
        self.min_elapsed_s = min_elapsed_s
        self.min_remaining_s = min_remaining_s
        self.min_up_prob = min_up_prob
        self.max_up_prob = max_up_prob
        self._last_btc: float | None = None

    def decide(self, state: DecisionState) -> StrategyDecision:
        tick = state.tick

        if tick.up_price is None or tick.down_price is None:
            return StrategyDecision("hold", "missing polymarket price")

        if tick.btc_chainlink is None and tick.btc_binance is None:
            return StrategyDecision("hold", "missing btc price")

        btc = tick.btc_chainlink if tick.btc_chainlink is not None else tick.btc_binance

        if tick.elapsed is not None and tick.elapsed < self.min_elapsed_s:
            self._last_btc = btc
            return StrategyDecision("hold", "too early")

        if tick.seconds_left is not None and tick.seconds_left < self.min_remaining_s:
            self._last_btc = btc
            return StrategyDecision("hold", "too late")

        if not (self.min_up_prob <= tick.up_price <= self.max_up_prob):
            self._last_btc = btc
            return StrategyDecision("hold", "up price outside allowed range")

        if self._last_btc is None:
            self._last_btc = btc
            return StrategyDecision("hold", "first btc value")

        pct_change = ((btc - self._last_btc) / self._last_btc) * 100.0
        self._last_btc = btc

        if pct_change >= self.momentum_pct:
            return StrategyDecision("buy_up", f"btc momentum +{pct_change:.4f}%")

        if pct_change <= -self.momentum_pct:
            return StrategyDecision("buy_down", f"btc momentum {pct_change:.4f}%")

        return StrategyDecision("hold", f"no signal {pct_change:+.4f}%")


class RuleBasedStrategy(BaseStrategy):
    name = "rule_based"

    def __init__(
        self,
        rules: list[dict[str, Any]],
        default_usd_amount: float = 1.0,
        max_orders: int | None = None,
        cooldown_ticks: int = 0,
    ):
        self.rules = rules
        self.default_usd_amount = default_usd_amount
        self.max_orders = max_orders
        self.cooldown_ticks = cooldown_ticks
        self._last_up_price: float | None = None
        self._last_down_price: float | None = None
        self._last_btc: float | None = None
        self._ticks_since_trade = cooldown_ticks

    def decide(self, state: DecisionState) -> StrategyDecision:
        metrics = self._build_metrics(state)

        if self._ticks_since_trade < self.cooldown_ticks:
            self._ticks_since_trade += 1
            self._remember(metrics)
            return StrategyDecision("hold", "cooldown")

        for rule in self.rules:
            if self._rule_matches(rule, metrics):
                self._ticks_since_trade = 0
                self._remember(metrics)
                action = str(rule.get("action", "hold")).lower().strip()
                if action in {"buy_up", "buy_down"} and self.max_orders is not None and state.orders_placed >= self.max_orders:
                    return StrategyDecision("hold", "max buy orders reached")
                usd_amount = rule.get("usd_amount", self.default_usd_amount)
                return StrategyDecision(
                    action=action,
                    reason=str(rule.get("name", f"rule matched: {action}")),
                    usd_amount=float(usd_amount) if usd_amount is not None else None,
                )

        self._ticks_since_trade += 1
        self._remember(metrics)
        return StrategyDecision("hold", "no rule matched")

    def _build_metrics(self, state: DecisionState) -> dict[str, float | bool | None]:
        tick = state.tick
        btc = tick.btc_chainlink if tick.btc_chainlink is not None else tick.btc_binance

        return {
            "up_price": tick.up_price,
            "down_price": tick.down_price,
            "up_minus_down_price": distance(tick.up_price, tick.down_price),
            "down_minus_up_price": distance(tick.down_price, tick.up_price),
            "up_price_distance_from_even": distance(tick.up_price, 0.5),
            "down_price_distance_from_even": distance(tick.down_price, 0.5),
            "up_price_pct_change": pct_change(tick.up_price, self._last_up_price),
            "down_price_pct_change": pct_change(tick.down_price, self._last_down_price),
            "btc_price": btc,
            "btc_pct_change": pct_change(btc, self._last_btc),
            "btc_distance_to_price_to_beat": distance(btc, tick.price_to_beat),
            "btc_distance_to_price_to_beat_pct": distance_pct(btc, tick.price_to_beat),
            "abs_btc_distance_to_price_to_beat": abs_distance(btc, tick.price_to_beat),
            "abs_btc_distance_to_price_to_beat_pct": abs_distance_pct(btc, tick.price_to_beat),
            "btc_above_price_to_beat": btc is not None and tick.price_to_beat is not None and btc >= tick.price_to_beat,
            "btc_below_price_to_beat": btc is not None and tick.price_to_beat is not None and btc < tick.price_to_beat,
            "seconds_left": tick.seconds_left,
            "elapsed": tick.elapsed,
            "cash": state.cash,
            "current_balance": state.current_balance,
            "up_tokens": state.up_tokens,
            "down_tokens": state.down_tokens,
            "has_up_position": state.up_tokens > 0,
            "has_down_position": state.down_tokens > 0,
            "up_position_value": state.up_tokens * tick.up_price if tick.up_price is not None else None,
            "down_position_value": state.down_tokens * tick.down_price if tick.down_price is not None else None,
            "orders_placed": state.orders_placed,
        }

    def _remember(self, metrics: dict[str, float | bool | None]) -> None:
        self._last_up_price = as_float(metrics.get("up_price"))
        self._last_down_price = as_float(metrics.get("down_price"))
        self._last_btc = as_float(metrics.get("btc_price"))

    def _rule_matches(self, rule: dict[str, Any], metrics: dict[str, float | bool | None]) -> bool:
        if "all" in rule:
            return all(condition_matches(condition, metrics) for condition in rule["all"])

        if "any" in rule:
            return any(condition_matches(condition, metrics) for condition in rule["any"])

        return condition_matches(rule.get("when", rule), metrics)


def condition_matches(condition: dict[str, Any], metrics: dict[str, float | bool | None]) -> bool:
    metric_name = condition.get("metric")
    operator = str(condition.get("operator", "==")).lower().strip()
    expected = condition.get("value")

    if metric_name not in metrics:
        raise ValueError(f"Unknown rule metric: {metric_name!r}")

    actual = metrics[metric_name]
    if actual is None:
        return False

    if operator in {"is", "==", "="}:
        return actual == expected
    if operator in {"!=", "not"}:
        return actual != expected

    actual_float = as_float(actual)
    expected_float = as_float(expected)
    if actual_float is None or expected_float is None:
        return False

    if operator == ">":
        return actual_float > expected_float
    if operator == ">=":
        return actual_float >= expected_float
    if operator == "<":
        return actual_float < expected_float
    if operator == "<=":
        return actual_float <= expected_float

    raise ValueError(f"Unknown rule operator: {operator!r}")


def as_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pct_change(current: float | None, previous: float | None) -> float | None:
    if current is None or previous is None or previous == 0:
        return None
    return ((current - previous) / previous) * 100.0


def distance(value: float | None, target: float | None) -> float | None:
    if value is None or target is None:
        return None
    return value - target


def distance_pct(value: float | None, target: float | None) -> float | None:
    if value is None or target is None or target == 0:
        return None
    return ((value - target) / target) * 100.0


def abs_distance(value: float | None, target: float | None) -> float | None:
    raw_distance = distance(value, target)
    if raw_distance is None:
        return None
    return abs(raw_distance)


def abs_distance_pct(value: float | None, target: float | None) -> float | None:
    raw_distance_pct = distance_pct(value, target)
    if raw_distance_pct is None:
        return None
    return abs(raw_distance_pct)
