import unittest

from simulator.models import DecisionState, MarketTick
from simulator.strategies import RuleBasedStrategy


def make_state(*, market_start_balance=200.0, up_price=0.25, down_price=0.75):
    tick = MarketTick(
        timestamp="2026-05-30T00:00:00Z",
        unix_time=0.0,
        seconds_left=120.0,
        elapsed=10.0,
        up_price=up_price,
        down_price=down_price,
        btc_binance=100.0,
        btc_chainlink=None,
        price_to_beat=99.0,
    )
    return DecisionState(
        tick=tick,
        cash=market_start_balance,
        up_tokens=0.0,
        down_tokens=0.0,
        current_balance=market_start_balance,
        market_start_balance=market_start_balance,
        market_spend_used=0.0,
        last_action="none",
        orders_placed=0,
    )


class RuleBasedStrategySizingTests(unittest.TestCase):
    def test_balance_pct_sizes_buy_from_market_start_balance(self):
        strategy = RuleBasedStrategy(
            rules=[
                {
                    "name": "buy_pct",
                    "when": {"metric": "up_price", "operator": "<=", "value": 0.5},
                    "action": "buy_up",
                    "balance_pct": 0.10,
                }
            ],
            default_usd_amount=1.0,
        )

        decision = strategy.decide(make_state(market_start_balance=200.0))

        self.assertEqual(decision.action, "buy_up")
        self.assertEqual(decision.reason, "buy_pct")
        self.assertEqual(decision.usd_amount, 20.0)

    def test_token_amount_still_overrides_balance_pct(self):
        strategy = RuleBasedStrategy(
            rules=[
                {
                    "name": "buy_tokens",
                    "when": {"metric": "up_price", "operator": "<=", "value": 0.5},
                    "action": "buy_up",
                    "balance_pct": 0.10,
                    "token_amount": 12,
                }
            ],
            default_usd_amount=1.0,
        )

        decision = strategy.decide(make_state(market_start_balance=200.0, up_price=0.25))

        self.assertEqual(decision.usd_amount, 3.0)


if __name__ == "__main__":
    unittest.main()
