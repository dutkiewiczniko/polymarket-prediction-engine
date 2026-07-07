import unittest

from simulator.models import DecisionState, MarketTick
from simulator.strategies import BaseStrategy, StrategyDecision, VotingEnsembleStrategy


def make_state() -> DecisionState:
    tick = MarketTick(
        timestamp="2026-07-07T00:00:00Z",
        unix_time=0.0,
        seconds_left=120.0,
        elapsed=10.0,
        up_price=0.5,
        down_price=0.5,
        btc_binance=100.0,
        btc_chainlink=None,
        price_to_beat=99.0,
    )
    return DecisionState(
        tick=tick,
        cash=100.0,
        up_tokens=0.0,
        down_tokens=0.0,
        current_balance=100.0,
        market_start_balance=100.0,
        market_spend_used=0.0,
        last_action="none",
        orders_placed=0,
    )


class FixedStrategy(BaseStrategy):
    def __init__(self, decision: StrategyDecision, name: str):
        self._decision = decision
        self.name = name

    def decide(self, state: DecisionState) -> StrategyDecision:
        return self._decision


class VotingEnsembleSizingTests(unittest.TestCase):
    def make_ensemble(self, *, size_mode: str = "min") -> VotingEnsembleStrategy:
        members = [
            FixedStrategy(StrategyDecision("buy_up", "small", usd_amount=5.0), "small_buyer"),
            FixedStrategy(StrategyDecision("buy_up", "large", usd_amount=10.0), "large_buyer"),
        ]
        return VotingEnsembleStrategy(
            members=members,
            member_names=[m.name for m in members],
            min_votes=2,
            default_scale=1.0,
            size_mode=size_mode,
        )

    def test_default_size_mode_is_min(self):
        ensemble = VotingEnsembleStrategy(
            members=[FixedStrategy(StrategyDecision("hold"), "m")],
            member_names=["m"],
        )
        self.assertEqual(ensemble.size_mode, "min")

    def test_min_size_mode_takes_smallest_agreeing_vote(self):
        ensemble = self.make_ensemble(size_mode="min")
        decision = ensemble.decide(make_state())
        self.assertEqual(decision.action, "buy_up")
        self.assertEqual(decision.usd_amount, 5.0)

    def test_max_size_mode_takes_largest_agreeing_vote(self):
        ensemble = self.make_ensemble(size_mode="max")
        decision = ensemble.decide(make_state())
        self.assertEqual(decision.action, "buy_up")
        self.assertEqual(decision.usd_amount, 10.0)

    def test_invalid_size_mode_rejected(self):
        with self.assertRaises(ValueError):
            VotingEnsembleStrategy(
                members=[FixedStrategy(StrategyDecision("hold"), "m")],
                member_names=["m"],
                size_mode="average",
            )


if __name__ == "__main__":
    unittest.main()
