import unittest

from simulator.models import DecisionState, MarketTick
from simulator.strategies import RuleBasedStrategy


def make_state(*, up_price=0.30, down_price=0.70, btc_price=100.0, unix_time=0.0):
    tick = MarketTick(
        timestamp="2026-07-08T00:00:00Z",
        unix_time=unix_time,
        seconds_left=120.0,
        elapsed=10.0,
        up_price=up_price,
        down_price=down_price,
        btc_binance=btc_price,
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


def two_rule_strategy(scope: str) -> RuleBasedStrategy:
    return RuleBasedStrategy(
        rules=[
            {
                "name": "buy_up_rule",
                "when": {"metric": "up_price", "operator": "<=", "value": 0.5},
                "action": "buy_up",
                "usd_amount": 2.0,
            },
            {
                "name": "buy_down_rule",
                "when": {"metric": "down_price", "operator": ">=", "value": 0.5},
                "action": "buy_down",
                "usd_amount": 3.0,
            },
        ],
        cooldown_ticks=3,
        cooldown_scope=scope,
    )


class GlobalCooldownTests(unittest.TestCase):
    def test_global_scope_blocks_all_rules_after_any_fire(self):
        strategy = two_rule_strategy("global")

        first = strategy.decide(make_state())
        self.assertEqual(first.action, "buy_up")

        # Both rules' conditions still hold, but the global cooldown blocks everything.
        second = strategy.decide(make_state())
        self.assertEqual(second.action, "hold")
        self.assertEqual(second.reason, "cooldown")

    def test_global_is_the_default_scope(self):
        strategy = RuleBasedStrategy(rules=[])
        self.assertEqual(strategy.cooldown_scope, "global")


class PerRuleCooldownTests(unittest.TestCase):
    def test_cooling_rule_does_not_block_other_rules(self):
        strategy = two_rule_strategy("per_rule")

        first = strategy.decide(make_state())
        self.assertEqual(first.action, "buy_up")

        # buy_up_rule is cooling down; buy_down_rule still gets its turn.
        second = strategy.decide(make_state())
        self.assertEqual(second.action, "buy_down")
        self.assertEqual(second.reason, "buy_down_rule")

    def test_rule_fires_again_after_its_own_cooldown_expires(self):
        strategy = two_rule_strategy("per_rule")

        actions = [strategy.decide(make_state()).action for _ in range(6)]
        # tick1: up fires; tick2: down fires; ticks 3-4: both cooling -> hold;
        # tick5: up available again (3-tick cooldown elapsed); tick6: down again.
        self.assertEqual(actions, ["buy_up", "buy_down", "hold", "hold", "buy_up", "buy_down"])

    def test_exhausted_pool_falls_through_to_next_rule(self):
        strategy = RuleBasedStrategy(
            rules=[
                {
                    "name": "pooled_buy_up",
                    "pool": "tiny",
                    "when": {"metric": "up_price", "operator": "<=", "value": 0.5},
                    "action": "buy_up",
                    "usd_amount": 5.0,
                },
                {
                    "name": "fallback_buy_down",
                    "when": {"metric": "down_price", "operator": ">=", "value": 0.5},
                    "action": "buy_down",
                    "usd_amount": 1.0,
                },
            ],
            cooldown_ticks=0,
            cooldown_scope="per_rule",
            pools={"tiny": {"max_pool_spend_usd": 5.0}},
        )

        first = strategy.decide(make_state())
        self.assertEqual(first.action, "buy_up")

        # Pool now exhausted: in per_rule scope the matched rule is skipped and
        # the next rule still evaluates, instead of a blocking hold.
        second = strategy.decide(make_state())
        self.assertEqual(second.action, "buy_down")
        self.assertEqual(second.reason, "fallback_buy_down")

    def test_hold_rules_still_act_as_guards(self):
        strategy = RuleBasedStrategy(
            rules=[
                {
                    "name": "suppress_zone",
                    "when": {"metric": "up_price", "operator": "<=", "value": 0.5},
                    "action": "hold",
                },
                {
                    "name": "buy_down_rule",
                    "when": {"metric": "down_price", "operator": ">=", "value": 0.5},
                    "action": "buy_down",
                    "usd_amount": 1.0,
                },
            ],
            cooldown_ticks=0,
            cooldown_scope="per_rule",
        )

        decision = strategy.decide(make_state())
        self.assertEqual(decision.action, "hold")
        self.assertEqual(decision.reason, "suppress_zone")

    def test_invalid_scope_rejected(self):
        with self.assertRaises(ValueError):
            RuleBasedStrategy(rules=[], cooldown_scope="per_market")


if __name__ == "__main__":
    unittest.main()
