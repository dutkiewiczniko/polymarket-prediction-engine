\
import random
from dataclasses import dataclass
from simulator.models import DecisionState


@dataclass
class StrategyDecision:
    action: str
    reason: str = ""


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
