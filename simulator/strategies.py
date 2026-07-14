\
import bisect
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

    def seed_btc_history(self, samples: list[tuple[float, float]]) -> None:
        """Pre-load (unix_time, btc_price) samples from before this market opened,
        so time-window momentum metrics have history at market start instead of
        returning None until the window has elapsed in-market. BTC is one
        continuous price stream, so prior-market samples are valid history.
        Token (up/down) prices must NOT be seeded this way -- they belong to a
        different market and reset each round.

        NOTE: backtest-only for now. The live engine (scripts/live/
        live_strategy_suite.py) rebuilds strategies per market and does not yet
        carry BTC history across the boundary -- that must be implemented there
        before running any long-window (>60s) momentum strategy live.
        """


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

    # Windows (seconds) for the momentum_*s / up_price_momentum_*s / down_price_momentum_*s
    # metrics. Tick spacing in recorded markets is well under 1 second, so these look back
    # by elapsed time (like the offline btc_return_Xs features in build_training_market_data.py),
    # not by tick count -- a fixed N-tick lookback is not a stable time window.
    # Windows above 300s exceed one market's lifetime and can only ever have data
    # via prior-market history seeding (seed_btc_history) -- they are permanently
    # None in cold runs and in the live engine until warmup is ported there.
    MOMENTUM_WINDOWS_S = (
        1.0, 2.0, 3.0, 5.0, 10.0, 15.0, 20.0, 30.0, 40.0, 60.0, 80.0, 120.0, 160.0,
        230.0, 240.0, 250.0, 275.0, 300.0, 420.0, 600.0, 900.0, 1200.0, 1500.0,
    )
    _MOMENTUM_HISTORY_BUFFER_S = max(MOMENTUM_WINDOWS_S) + 10.0

    # momentum_Nt / up_price_momentum_Nt / down_price_momentum_Nt look back exactly
    # N recorded ticks (not elapsed seconds) -- since tick spacing is sub-second,
    # this is a much faster-reacting (and noisier) signal than momentum_1s.
    MOMENTUM_TICK_WINDOWS = (1, 2, 3, 4)

    def __init__(
        self,
        rules: list[dict[str, Any]],
        default_usd_amount: float = 1.0,
        max_orders: int | None = None,
        cooldown_ticks: int = 0,
        sell_opposite_first: bool = True,
        max_market_spend_usd: float | None = None,
        combine_matching_buys: bool = False,
        pools: dict[str, dict[str, Any]] | None = None,
        cooldown_scope: str = "global",
        compute_all_metrics: bool = False,
    ):
        if cooldown_scope not in {"global", "per_rule"}:
            raise ValueError(f"Unknown cooldown_scope: {cooldown_scope!r}")
        self.rules = rules
        self.default_usd_amount = default_usd_amount
        self.max_orders = max_orders
        self.cooldown_ticks = cooldown_ticks
        self.sell_opposite_first = sell_opposite_first
        self.max_market_spend_usd = max_market_spend_usd
        self.combine_matching_buys = combine_matching_buys
        self.pools = pools or {}
        self.cooldown_scope = cooldown_scope
        self.compute_all_metrics = compute_all_metrics
        self._last_up_price: float | None = None
        self._last_down_price: float | None = None
        self._last_btc: float | None = None
        self._ticks_since_trade = cooldown_ticks
        self._active_cooldown_ticks = cooldown_ticks
        self._pool_spend: dict[str, float] = {name: 0.0 for name in self.pools}
        self._tick_index = 0
        self._rule_last_action_tick: dict[int, int] = {}
        self._up_series: list[tuple[float, float]] = []
        self._down_series: list[tuple[float, float]] = []
        self._btc_series: list[tuple[float, float]] = []

        # Only compute the momentum windows this config's rules actually
        # reference. Every tick otherwise builds all 23 time-windows x 3 series +
        # 4 tick-windows x 3 series = 81 momentum values, ~95% of which a typical
        # config never reads. The metric names below are looked up by rule
        # conditions only, so filtering to the referenced set is decision-identical
        # (an unreferenced momentum metric can never affect a rule). Long windows
        # (e.g. 900s) are O(log n) bisect regardless of length -- the win is
        # skipping unused windows, not skipping long ones.
        self._used_metrics = self._collect_used_metrics()
        self._btc_time_windows = self._filter_windows("momentum_{0}s", self.MOMENTUM_WINDOWS_S)
        self._up_time_windows = self._filter_windows("up_price_momentum_{0}s", self.MOMENTUM_WINDOWS_S)
        self._down_time_windows = self._filter_windows("down_price_momentum_{0}s", self.MOMENTUM_WINDOWS_S)
        self._btc_tick_windows = self._filter_windows("momentum_{0}t", self.MOMENTUM_TICK_WINDOWS)
        self._up_tick_windows = self._filter_windows("up_price_momentum_{0}t", self.MOMENTUM_TICK_WINDOWS)
        self._down_tick_windows = self._filter_windows("down_price_momentum_{0}t", self.MOMENTUM_TICK_WINDOWS)
        self._track_up_series = bool(self._up_time_windows or self._up_tick_windows)
        self._track_down_series = bool(self._down_time_windows or self._down_tick_windows)

    def _collect_used_metrics(self) -> set[str]:
        """Metric names referenced by any rule condition (all / any / when / bare).
        Conditions are flat {metric, operator, value} dicts -- there is no nesting."""
        used: set[str] = set()

        def add(condition: Any) -> None:
            if isinstance(condition, dict):
                name = condition.get("metric")
                if isinstance(name, str):
                    used.add(name)

        for rule in self.rules:
            if "all" in rule:
                for condition in rule["all"]:
                    add(condition)
            elif "any" in rule:
                for condition in rule["any"]:
                    add(condition)
            else:
                add(rule.get("when", rule))
        return used

    def _filter_windows(self, name_template: str, windows) -> list[tuple[str, float | int]]:
        """(metric_name, window) pairs for only the windows this config references."""
        pairs = []
        for window in windows:
            name = name_template.format(int(window))
            if self.compute_all_metrics or name in self._used_metrics:
                pairs.append((name, window))
        return pairs

    def decide(self, state: DecisionState) -> StrategyDecision:
        metrics = self._build_metrics(state)

        if self.cooldown_scope == "per_rule":
            return self._decide_per_rule(state, metrics)

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
                pool_name = rule.get("pool")
                if action in {"buy_up", "buy_down"} and pool_name is not None:
                    if pool_name not in self.pools:
                        raise ValueError(f"Rule references unknown pool: {pool_name!r}")
                    if not risk_override:
                        pool_cap = self._resolve_pool_cap(pool_name, metrics)
                        if pool_cap is not None:
                            remaining_pool = pool_cap - self._pool_spend[pool_name]
                            if remaining_pool <= 0:
                                return StrategyDecision("hold", f"max pool spend reached ({pool_name})")
                            usd_amount = min(float(usd_amount), remaining_pool)
                    self._pool_spend[pool_name] += float(usd_amount)
                elif action in {"buy_up", "buy_down"} and self.max_market_spend_usd is not None and not risk_override:
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

    def _decide_per_rule(self, state: DecisionState, metrics: dict[str, float | bool | None]) -> StrategyDecision:
        """Non-interfering rule evaluation (cooldown_scope: per_rule).

        Differences from the global path, all aimed at rules not blocking each
        other: (1) each rule has its own cooldown, started only when the rule
        actually returns a buy/sell -- a matching rule that can't act does not
        freeze the strategy; (2) a rule that is cooling down, over max_orders,
        unsizeable, or out of pool budget is SKIPPED so lower rules still get
        evaluated, instead of returning a blocking hold. Explicit hold-action
        rules still return hold immediately -- they are deliberate guards.
        """
        self._tick_index += 1
        self._remember(metrics)

        for rule_index, rule in enumerate(self.rules):
            rule_cooldown = int(rule.get("cooldown_ticks", self.cooldown_ticks))
            last_action_tick = self._rule_last_action_tick.get(rule_index)
            if last_action_tick is not None and self._tick_index - last_action_tick <= rule_cooldown:
                continue
            if not self._rule_matches(rule, metrics):
                continue

            action = str(rule.get("action", "hold")).lower().strip()
            reason = str(rule.get("name", f"rule matched: {action}"))
            if action == "hold":
                return StrategyDecision("hold", reason)

            risk_override = self._rule_risk_override(rule, action, metrics)
            if (
                action in {"buy_up", "buy_down"}
                and self.max_orders is not None
                and state.orders_placed >= self.max_orders
                and not risk_override
            ):
                continue
            usd_amount = self._rule_usd_amount(rule, action, metrics)
            if isinstance(usd_amount, StrategyDecision):
                continue
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
            pool_name = rule.get("pool")
            if action in {"buy_up", "buy_down"} and pool_name is not None:
                if pool_name not in self.pools:
                    raise ValueError(f"Rule references unknown pool: {pool_name!r}")
                if not risk_override:
                    pool_cap = self._resolve_pool_cap(pool_name, metrics)
                    if pool_cap is not None:
                        remaining_pool = pool_cap - self._pool_spend[pool_name]
                        if remaining_pool <= 0:
                            continue
                        usd_amount = min(float(usd_amount), remaining_pool)
                self._pool_spend[pool_name] += float(usd_amount)
            elif action in {"buy_up", "buy_down"} and self.max_market_spend_usd is not None and not risk_override:
                remaining_spend = self.max_market_spend_usd - state.market_spend_used
                if remaining_spend <= 0:
                    continue
                usd_amount = min(float(usd_amount), remaining_spend)

            self._rule_last_action_tick[rule_index] = self._tick_index
            return StrategyDecision(
                action=action,
                reason=reason,
                usd_amount=float(usd_amount) if usd_amount is not None else None,
                sell_opposite_first=sell_opposite_first,
            )

        return StrategyDecision("hold", "no rule matched")

    def seed_btc_history(self, samples: list[tuple[float, float]]) -> None:
        for now_time, btc in samples:
            self._record_momentum_sample(self._btc_series, now_time, btc)

    def _record_momentum_sample(
        self, series: list[tuple[float, float]], now_time: float | None, value: float | None
    ) -> None:
        if now_time is None or value is None:
            return
        series.append((now_time, value))
        cutoff = now_time - self._MOMENTUM_HISTORY_BUFFER_S
        while len(series) > 1 and series[0][0] < cutoff:
            series.pop(0)

    def _build_metrics(self, state: DecisionState) -> dict[str, float | bool | None]:
        tick = state.tick
        btc = tick.btc_chainlink if tick.btc_chainlink is not None else tick.btc_binance
        now_time = tick.unix_time

        if self._track_up_series:
            self._record_momentum_sample(self._up_series, now_time, tick.up_price)
        if self._track_down_series:
            self._record_momentum_sample(self._down_series, now_time, tick.down_price)
        self._record_momentum_sample(self._btc_series, now_time, btc)

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
            # Binance-specific signed distance: resolution is Chainlink but
            # Binance leads it by seconds -- a cheap side that Binance has
            # already crossed toward is a look-ahead buy (the engine's blended
            # `btc` metric prefers Chainlink, hiding exactly that lead).
            "btc_binance_distance_to_price_to_beat_pct": distance_pct(tick.btc_binance, tick.price_to_beat),
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
        # Only the referenced momentum windows (see __init__). btc-derived windows
        # stay None when btc is missing, matching the previous dict-comprehensions.
        for key, window_s in self._btc_time_windows:
            metrics[key] = momentum_pct(self._btc_series, now_time, window_s) if btc is not None else None
        for key, window_s in self._up_time_windows:
            metrics[key] = momentum_pct(self._up_series, now_time, window_s)
        for key, window_s in self._down_time_windows:
            metrics[key] = momentum_pct(self._down_series, now_time, window_s)
        for key, ticks in self._btc_tick_windows:
            metrics[key] = momentum_pct_by_ticks(self._btc_series, ticks) if btc is not None else None
        for key, ticks in self._up_tick_windows:
            metrics[key] = momentum_pct_by_ticks(self._up_series, ticks)
        for key, ticks in self._down_tick_windows:
            metrics[key] = momentum_pct_by_ticks(self._down_series, ticks)
        for key, value in getattr(tick, "extra", {}).items():
            if key in metrics:
                continue
            parsed = as_float(value)
            metrics[key] = parsed if parsed is not None else value
        return metrics

    def _resolve_pool_cap(self, pool_name: str, metrics: dict[str, float | bool | None]) -> float | None:
        pool_cfg = self.pools[pool_name]
        caps = []
        usd_cap = as_float(pool_cfg.get("max_pool_spend_usd"))
        if usd_cap is not None:
            caps.append(usd_cap)
        pct_cap = as_float(pool_cfg.get("max_pool_spend_pct"))
        if pct_cap is not None:
            balance = as_float(metrics.get("market_start_balance"))
            if balance is not None:
                caps.append(pct_cap * balance)
        if not caps:
            return None
        return min(caps)

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
        size_mode: str = "min",
    ):
        if size_mode not in {"min", "max"}:
            raise ValueError(f"Unknown voting_ensemble size_mode: {size_mode!r}")
        self.members = members
        self.member_names = member_names
        self.min_votes = min_votes
        self.priority_members = set(priority_members or [])
        self.priority_scale = priority_scale
        self.default_scale = default_scale
        self.max_orders = max_orders
        self.cooldown_ticks = cooldown_ticks
        self.size_mode = size_mode
        self._ticks_since_trade = cooldown_ticks

    def seed_btc_history(self, samples: list[tuple[float, float]]) -> None:
        for member in self.members:
            member.seed_btc_history(samples)

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
        if sized_votes:
            size_pick = max(sized_votes) if self.size_mode == "max" else min(sized_votes)
            usd_amount = size_pick * scale
        else:
            usd_amount = None
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

    def seed_btc_history(self, samples: list[tuple[float, float]]) -> None:
        self.override.seed_btc_history(samples)
        self.base.seed_btc_history(samples)

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


def latest_value_at_or_before(series: list[tuple[float, float]], cutoff_time: float) -> float | None:
    """series must be sorted ascending by timestamp (true of the append-only,
    front-trimmed momentum sample buffers). Binary search instead of a linear
    backward scan -- this is called for every momentum window on every tick,
    so an O(n) scan here is the dominant cost of RuleBasedStrategy replay.
    """
    idx = bisect.bisect_right(series, (cutoff_time, float("inf")))
    if idx == 0:
        return None
    return series[idx - 1][1]


def momentum_pct(series: list[tuple[float, float]], now_time: float | None, window_s: float) -> float | None:
    """Percent change over the trailing `window_s` seconds of elapsed market time,
    looked up in a (unix_time, value) series -- not a fixed number of ticks back,
    since tick spacing in recorded markets is well under 1 second.
    """
    if not series or now_time is None:
        return None
    current = series[-1][1]
    previous = latest_value_at_or_before(series, now_time - window_s)
    return pct_change(current, previous)


def momentum_pct_by_ticks(series: list[tuple[float, float]], ticks_back: int) -> float | None:
    """Percent change vs. exactly `ticks_back` recorded ticks ago (not elapsed time).
    Much faster-reacting and noisier than momentum_pct, since tick spacing is
    sub-second -- "1 tick ago" is well under 1 real second.
    """
    if len(series) <= ticks_back:
        return None
    current = series[-1][1]
    previous = series[-1 - ticks_back][1]
    return pct_change(current, previous)
