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
    sell_opposite_first: bool | None = None


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
        sell_opposite_first: bool = True,
        max_market_spend_usd: float | None = None,
        combine_matching_buys: bool = False,
    ):
        self.rules = rules
        self.default_usd_amount = default_usd_amount
        self.max_orders = max_orders
        self.cooldown_ticks = cooldown_ticks
        self.sell_opposite_first = sell_opposite_first
        self.max_market_spend_usd = max_market_spend_usd
        self.combine_matching_buys = combine_matching_buys
        self._last_up_price: float | None = None
        self._last_down_price: float | None = None
        self._last_btc: float | None = None
        self._ticks_since_trade = cooldown_ticks
        self._active_cooldown_ticks = cooldown_ticks

    def decide(self, state: DecisionState) -> StrategyDecision:
        metrics = self._build_metrics(state)

        if self._ticks_since_trade < self._active_cooldown_ticks:
            self._ticks_since_trade += 1
            self._remember(metrics)
            return StrategyDecision("hold", "cooldown")

        for rule_index, rule in enumerate(self.rules):
            if self._rule_matches(rule, metrics):
                self._active_cooldown_ticks = int(rule.get("cooldown_ticks", self.cooldown_ticks))
                self._ticks_since_trade = 0
                self._remember(metrics)
                action = str(rule.get("action", "hold")).lower().strip()
                risk_override = self._rule_risk_override(rule, action, metrics)
                if (
                    action in {"buy_up", "buy_down"}
                    and self.max_orders is not None
                    and state.orders_placed >= self.max_orders
                    and not risk_override
                ):
                    return StrategyDecision("hold", "max buy orders reached")
                usd_amount = self._rule_usd_amount(rule, action, metrics)
                if isinstance(usd_amount, StrategyDecision):
                    return usd_amount
                reason = str(rule.get("name", f"rule matched: {action}"))
                sell_opposite_first = as_bool(rule.get("sell_opposite_first", self.sell_opposite_first))
                if self.combine_matching_buys and action in {"buy_up", "buy_down"}:
                    extra_amount, extra_reasons, extra_sell_opposite_first = self._combined_same_side_buys(
                        rules=self.rules[rule_index + 1:],
                        action=action,
                        metrics=metrics,
                    )
                    usd_amount += extra_amount
                    if extra_reasons:
                        reason = " + ".join([reason, *extra_reasons])
                    sell_opposite_first = sell_opposite_first and extra_sell_opposite_first
                if action in {"buy_up", "buy_down"} and self.max_market_spend_usd is not None and not risk_override:
                    remaining_spend = self.max_market_spend_usd - state.market_spend_used
                    if remaining_spend <= 0:
                        return StrategyDecision("hold", "max market spend reached")
                    usd_amount = min(float(usd_amount), remaining_spend)
                return StrategyDecision(
                    action=action,
                    reason=reason,
                    usd_amount=float(usd_amount) if usd_amount is not None else None,
                    sell_opposite_first=sell_opposite_first,
                )

        self._ticks_since_trade += 1
        self._remember(metrics)
        return StrategyDecision("hold", "no rule matched")

    def _build_metrics(self, state: DecisionState) -> dict[str, float | bool | None]:
        tick = state.tick
        btc = tick.btc_chainlink if tick.btc_chainlink is not None else tick.btc_binance

        metrics = {
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
            "market_start_balance": state.market_start_balance,
            "market_spend_used": state.market_spend_used,
            "up_tokens": state.up_tokens,
            "down_tokens": state.down_tokens,
            "has_up_position": state.up_tokens > 0,
            "has_down_position": state.down_tokens > 0,
            "up_position_value": state.up_tokens * tick.up_price if tick.up_price is not None else None,
            "down_position_value": state.down_tokens * tick.down_price if tick.down_price is not None else None,
            "orders_placed": state.orders_placed,
        }
        for key, value in getattr(tick, "extra", {}).items():
            if key in metrics:
                continue
            parsed = as_float(value)
            metrics[key] = parsed if parsed is not None else value
        return metrics

    def _remember(self, metrics: dict[str, float | bool | None]) -> None:
        self._last_up_price = as_float(metrics.get("up_price"))
        self._last_down_price = as_float(metrics.get("down_price"))
        self._last_btc = as_float(metrics.get("btc_price"))

    def _rule_usd_amount(
        self,
        rule: dict[str, Any],
        action: str,
        metrics: dict[str, float | bool | None],
    ) -> float | StrategyDecision:
        usd_amount = rule.get("usd_amount", self.default_usd_amount)
        if action in {"buy_up", "buy_down"} and rule.get("balance_pct") is not None:
            balance_pct = as_float(rule.get("balance_pct"))
            scale_balance = as_float(metrics.get("market_start_balance"))
            if balance_pct is None or scale_balance is None:
                return StrategyDecision("hold", "cannot size balance_pct order")
            usd_amount = scale_balance * balance_pct
        if action in {"buy_up", "buy_down"} and rule.get("token_amount") is not None:
            price_metric = "up_price" if action == "buy_up" else "down_price"
            price = as_float(metrics.get(price_metric))
            token_amount = as_float(rule.get("token_amount"))
            if price is None or token_amount is None:
                return StrategyDecision("hold", "cannot size token_amount order")
            usd_amount = token_amount * price
        if action in {"buy_up", "buy_down"} and rule.get("balance_scaled_token_amount") is not None:
            price_metric = "up_price" if action == "buy_up" else "down_price"
            price = as_float(metrics.get(price_metric))
            token_basis = as_float(rule.get("balance_scaled_token_amount"))
            scale_balance = as_float(metrics.get("market_start_balance"))
            if price is None or token_basis is None or scale_balance is None:
                return StrategyDecision("hold", "cannot size balance_scaled_token_amount order")
            if rule.get("balance_scale_cap") is not None:
                cap = as_float(rule.get("balance_scale_cap"))
                if cap is None:
                    return StrategyDecision("hold", "invalid balance_scale_cap")
                scale_balance = min(scale_balance, cap)
            token_amount = token_basis * scale_balance / 100.0
            usd_amount = token_amount * price
        return float(usd_amount)

    def _rule_risk_override(
        self,
        rule: dict[str, Any],
        action: str,
        metrics: dict[str, float | bool | None],
    ) -> bool:
        if action not in {"buy_up", "buy_down"}:
            return False
        if as_bool(rule.get("ignore_risk_limits", False)):
            return True
        max_price = as_float(rule.get("risk_override_max_price"))
        if max_price is None:
            return False
        price_metric = "up_price" if action == "buy_up" else "down_price"
        price = as_float(metrics.get(price_metric))
        return price is not None and price <= max_price

    def _combined_same_side_buys(
        self,
        rules: list[dict[str, Any]],
        action: str,
        metrics: dict[str, float | bool | None],
    ) -> tuple[float, list[str], bool]:
        total = 0.0
        reasons = []
        sell_opposite_first = True
        for rule in rules:
            rule_action = str(rule.get("action", "hold")).lower().strip()
            if rule_action != action:
                continue
            if not self._rule_matches(rule, metrics):
                continue
            usd_amount = self._rule_usd_amount(rule, rule_action, metrics)
            if isinstance(usd_amount, StrategyDecision):
                continue
            total += usd_amount
            reasons.append(str(rule.get("name", f"rule matched: {rule_action}")))
            sell_opposite_first = sell_opposite_first and as_bool(
                rule.get("sell_opposite_first", self.sell_opposite_first)
            )
        return total, reasons, sell_opposite_first

    def _rule_matches(self, rule: dict[str, Any], metrics: dict[str, float | bool | None]) -> bool:
        if "all" in rule:
            return all(condition_matches(condition, metrics) for condition in rule["all"])

        if "any" in rule:
            return any(condition_matches(condition, metrics) for condition in rule["any"])

        return condition_matches(rule.get("when", rule), metrics)


class VotingEnsembleStrategy(BaseStrategy):
    name = "voting_ensemble"

    def __init__(
        self,
        members: list[BaseStrategy],
        member_names: list[str],
        min_votes: int = 2,
        priority_members: list[str] | None = None,
        priority_scale: float = 0.75,
        default_scale: float = 0.75,
        max_orders: int | None = None,
        cooldown_ticks: int = 0,
    ):
        self.members = members
        self.member_names = member_names
        self.min_votes = min_votes
        self.priority_members = set(priority_members or [])
        self.priority_scale = priority_scale
        self.default_scale = default_scale
        self.max_orders = max_orders
        self.cooldown_ticks = cooldown_ticks
        self._ticks_since_trade = cooldown_ticks

    def decide(self, state: DecisionState) -> StrategyDecision:
        member_decisions = [
            (name, strategy.decide(state))
            for name, strategy in zip(self.member_names, self.members)
        ]

        if self._ticks_since_trade < self.cooldown_ticks:
            self._ticks_since_trade += 1
            return StrategyDecision("hold", self._format_reason("cooldown", member_decisions))

        if self.max_orders is not None and state.orders_placed >= self.max_orders:
            return StrategyDecision("hold", self._format_reason("max buy orders reached", member_decisions))

        buy_decisions = [
            (name, decision)
            for name, decision in member_decisions
            if decision.action in {"buy_up", "buy_down"}
        ]
        up_votes = [(name, decision) for name, decision in buy_decisions if decision.action == "buy_up"]
        down_votes = [(name, decision) for name, decision in buy_decisions if decision.action == "buy_down"]

        chosen = self._choose_side(up_votes, down_votes)
        if chosen is None:
            return StrategyDecision("hold", self._format_reason("insufficient aligned votes", member_decisions))

        action, votes = chosen
        scale = self.priority_scale if any(name in self.priority_members for name, _ in votes) else self.default_scale
        sized_votes = [
            decision.usd_amount
            for _, decision in votes
            if decision.usd_amount is not None and decision.usd_amount > 0
        ]
        usd_amount = min(sized_votes) * scale if sized_votes else None
        sell_opposite_first = not any(decision.sell_opposite_first is False for _, decision in votes)
        self._ticks_since_trade = 0
        voters = ",".join(name for name, _ in votes)
        return StrategyDecision(
            action,
            f"ensemble {action} votes={len(votes)} voters={voters}",
            usd_amount=usd_amount,
            sell_opposite_first=sell_opposite_first,
        )

    def _choose_side(
        self,
        up_votes: list[tuple[str, StrategyDecision]],
        down_votes: list[tuple[str, StrategyDecision]],
    ) -> tuple[str, list[tuple[str, StrategyDecision]]] | None:
        valid = []
        if len(up_votes) >= self.min_votes:
            valid.append(("buy_up", up_votes))
        if len(down_votes) >= self.min_votes:
            valid.append(("buy_down", down_votes))
        if not valid:
            return None

        priority_valid = [
            item for item in valid
            if any(name in self.priority_members for name, _ in item[1])
        ]
        candidates = priority_valid or valid
        candidates.sort(key=lambda item: (len(item[1]), self._priority_count(item[1])), reverse=True)
        return candidates[0]

    def _priority_count(self, votes: list[tuple[str, StrategyDecision]]) -> int:
        return sum(1 for name, _ in votes if name in self.priority_members)

    def _format_reason(self, prefix: str, member_decisions: list[tuple[str, StrategyDecision]]) -> str:
        active = [
            f"{name}:{decision.action}"
            for name, decision in member_decisions
            if decision.action != "hold"
        ]
        suffix = "; ".join(active[:6]) if active else "no active member signals"
        return f"{prefix}; {suffix}"


class OverrideStrategy(BaseStrategy):
    name = "override"

    def __init__(
        self,
        override: BaseStrategy,
        base: BaseStrategy,
        override_name: str = "override",
        base_name: str = "base",
        suppress_base_while_override_position: bool = False,
        suppress_base_when_override_near_price: float | None = None,
    ):
        self.override = override
        self.base = base
        self.override_name = override_name
        self.base_name = base_name
        self.suppress_base_while_override_position = suppress_base_while_override_position
        self.suppress_base_when_override_near_price = suppress_base_when_override_near_price
        self._override_position_side: str | None = None

    def decide(self, state: DecisionState) -> StrategyDecision:
        self._refresh_override_position(state)
        override_decision = self.override.decide(state)
        if override_decision.action != "hold":
            if override_decision.action == "buy_up":
                self._override_position_side = "up"
            elif override_decision.action == "buy_down":
                self._override_position_side = "down"
            return StrategyDecision(
                action=override_decision.action,
                reason=f"{self.override_name} override: {override_decision.reason}",
                usd_amount=override_decision.usd_amount,
                sell_opposite_first=override_decision.sell_opposite_first,
            )

        if self.suppress_base_while_override_position and self._override_position_side:
            return StrategyDecision(
                "hold",
                f"{self.override_name} {_side_label(self._override_position_side)} position active; suppress {self.base_name}",
            )

        if self._near_override_price(state):
            return StrategyDecision(
                "hold",
                f"{self.override_name} near cheap threshold; suppress {self.base_name}",
            )

        base_decision = self.base.decide(state)
        if base_decision.action != "hold":
            return StrategyDecision(
                action=base_decision.action,
                reason=f"{self.base_name}: {base_decision.reason}",
                usd_amount=base_decision.usd_amount,
                sell_opposite_first=base_decision.sell_opposite_first,
            )
        return StrategyDecision(
            "hold",
            f"{self.override_name}: {override_decision.reason}; {self.base_name}: {base_decision.reason}",
        )

    def _refresh_override_position(self, state: DecisionState) -> None:
        if self._override_position_side == "up" and state.up_tokens <= 0:
            self._override_position_side = None
        elif self._override_position_side == "down" and state.down_tokens <= 0:
            self._override_position_side = None

    def _near_override_price(self, state: DecisionState) -> bool:
        threshold = self.suppress_base_when_override_near_price
        if threshold is None:
            return False
        tick = state.tick
        return (
            (tick.up_price is not None and tick.up_price <= threshold)
            or (tick.down_price is not None and tick.down_price <= threshold)
        )


def _side_label(side: str) -> str:
    return "UP" if side == "up" else "DOWN"


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


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.lower().strip()
        if normalized in {"true", "1", "yes", "y", "on"}:
            return True
        if normalized in {"false", "0", "no", "n", "off"}:
            return False
    return bool(value)


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
