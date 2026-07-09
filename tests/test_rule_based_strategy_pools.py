import unittest

from simulator.models import DecisionState, MarketTick
from simulator.strategies import RuleBasedStrategy


def make_state(*, up_price=0.02, down_price=0.98, btc_price=100.0, price_to_beat=99.0):
    tick = MarketTick(
        timestamp="2026-07-07T00:00:00Z",
        unix_time=0.0,
        seconds_left=120.0,
        elapsed=10.0,
        up_price=up_price,
        down_price=down_price,
        btc_binance=btc_price,
        btc_chainlink=None,
        price_to_beat=price_to_beat,
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


class RuleBasedStrategyPoolTests(unittest.TestCase):
    def test_pool_caps_total_spend_across_repeated_matches(self):
        strategy = RuleBasedStrategy(
            rules=[
                {
                    "name": "cheap_up_lottery",
                    "pool": "lottery",
                    "when": {"metric": "up_price", "operator": "<=", "value": 0.025},
                    "action": "buy_up",
                    "usd_amount": 10.0,
                }
            ],
            cooldown_ticks=0,
            pools={"lottery": {"max_pool_spend_usd": 25.0}},
        )

        first = strategy.decide(make_state())
        second = strategy.decide(make_state())
        third = strategy.decide(make_state())

        self.assertEqual(first.usd_amount, 10.0)
        self.assertEqual(second.usd_amount, 10.0)
        # Only $5 left in the pool after two $10 buys -- gets capped down, not rejected.
        self.assertEqual(third.usd_amount, 5.0)

        fourth = strategy.decide(make_state())
        self.assertEqual(fourth.action, "hold")
        self.assertEqual(fourth.reason, "max pool spend reached (lottery)")

    def test_pools_are_independent(self):
        strategy = RuleBasedStrategy(
            rules=[
                {
                    "name": "cheap_up_lottery",
                    "pool": "lottery",
                    "when": {"metric": "up_price", "operator": "<=", "value": 0.025},
                    "action": "buy_up",
                    "usd_amount": 30.0,
                },
                {
                    "name": "chase_down",
                    "pool": "main",
                    "when": {"metric": "down_price", "operator": ">=", "value": 0.5},
                    "action": "buy_down",
                    "usd_amount": 40.0,
                },
            ],
            cooldown_ticks=0,
            pools={"lottery": {"max_pool_spend_usd": 20.0}, "main": {"max_pool_spend_usd": 70.0}},
        )

        # Lottery rule matches first (up_price <= 0.025) and is capped to its own $20 pool.
        decision = strategy.decide(make_state())
        self.assertEqual(decision.action, "buy_up")
        self.assertEqual(decision.usd_amount, 20.0)

        # Main pool is untouched by the lottery spend.
        self.assertEqual(strategy._pool_spend["main"], 0.0)
        self.assertEqual(strategy._pool_spend["lottery"], 20.0)

    def test_ignore_risk_limits_bypasses_pool_cap(self):
        strategy = RuleBasedStrategy(
            rules=[
                {
                    "name": "cheap_up_lottery",
                    "pool": "lottery",
                    "when": {"metric": "up_price", "operator": "<=", "value": 0.025},
                    "action": "buy_up",
                    "usd_amount": 50.0,
                    "ignore_risk_limits": True,
                }
            ],
            cooldown_ticks=0,
            pools={"lottery": {"max_pool_spend_usd": 20.0}},
        )

        decision = strategy.decide(make_state())
        self.assertEqual(decision.usd_amount, 50.0)

    def test_unknown_pool_raises(self):
        strategy = RuleBasedStrategy(
            rules=[
                {
                    "name": "cheap_up_lottery",
                    "pool": "does_not_exist",
                    "when": {"metric": "up_price", "operator": "<=", "value": 0.025},
                    "action": "buy_up",
                    "usd_amount": 10.0,
                }
            ],
            cooldown_ticks=0,
            pools={"lottery": {"max_pool_spend_usd": 20.0}},
        )

        with self.assertRaises(ValueError):
            strategy.decide(make_state())

    def test_pct_pool_cap_scales_with_market_start_balance(self):
        strategy = RuleBasedStrategy(
            rules=[
                {
                    "name": "cheap_up_lottery",
                    "pool": "lottery",
                    "when": {"metric": "up_price", "operator": "<=", "value": 0.025},
                    "action": "buy_up",
                    "usd_amount": 30.0,
                }
            ],
            cooldown_ticks=0,
            pools={"lottery": {"max_pool_spend_pct": 0.2}},
        )

        # make_state() uses market_start_balance=100.0 -> 20% cap = $20.
        decision = strategy.decide(make_state())
        self.assertEqual(decision.usd_amount, 20.0)

    def test_pct_and_usd_caps_take_the_stricter_minimum(self):
        strategy = RuleBasedStrategy(
            rules=[
                {
                    "name": "cheap_up_lottery",
                    "pool": "lottery",
                    "when": {"metric": "up_price", "operator": "<=", "value": 0.025},
                    "action": "buy_up",
                    "usd_amount": 30.0,
                }
            ],
            cooldown_ticks=0,
            # pct implies a $50 cap, usd caps it lower at $15 -- the stricter one should win.
            pools={"lottery": {"max_pool_spend_pct": 0.5, "max_pool_spend_usd": 15.0}},
        )

        decision = strategy.decide(make_state())
        self.assertEqual(decision.usd_amount, 15.0)

    def test_rule_without_pool_still_uses_legacy_global_cap(self):
        strategy = RuleBasedStrategy(
            rules=[
                {
                    "name": "cheap_up_lottery",
                    "when": {"metric": "up_price", "operator": "<=", "value": 0.025},
                    "action": "buy_up",
                    "usd_amount": 10.0,
                }
            ],
            cooldown_ticks=0,
            max_market_spend_usd=5.0,
            pools={"lottery": {"max_pool_spend_usd": 20.0}},
        )

        decision = strategy.decide(make_state())
        self.assertEqual(decision.usd_amount, 5.0)


if __name__ == "__main__":
    unittest.main()
