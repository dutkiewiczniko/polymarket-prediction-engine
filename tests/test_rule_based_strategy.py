import unittest

from simulator.execution import execute_action
from simulator.models import DecisionState, MarketTick
from simulator.portfolio import Portfolio
from simulator.strategies import RuleBasedStrategy


def make_state(*, market_start_balance=200.0, up_price=0.25, down_price=0.75, btc_price=100.0, price_to_beat=99.0):
    tick = MarketTick(
        timestamp="2026-05-30T00:00:00Z",
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


class RuleBasedStrategyExecutionOptionTests(unittest.TestCase):
    def test_rule_can_preserve_opposite_side_position(self):
        strategy = RuleBasedStrategy(
            rules=[
                {
                    "name": "cheap_down_without_closing_up",
                    "when": {"metric": "down_price", "operator": "<=", "value": 0.75},
                    "action": "buy_down",
                    "usd_amount": 10,
                    "sell_opposite_first": False,
                }
            ],
        )
        decision = strategy.decide(make_state(up_price=0.25, down_price=0.75, btc_price=98.0, price_to_beat=99.0))
        portfolio = Portfolio(cash=100.0, up_tokens=20.0)

        events = execute_action(
            portfolio=portfolio,
            action=decision.action,
            timestamp="2026-05-30T00:00:00Z",
            up_price=0.25,
            down_price=0.75,
            usd_amount=decision.usd_amount,
            sell_opposite_first=decision.sell_opposite_first,
            reason=decision.reason,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].action, "buy")
        self.assertEqual(events[0].side, "down")
        self.assertEqual(portfolio.up_tokens, 20.0)
        self.assertGreater(portfolio.down_tokens, 0.0)

    def test_default_buy_closes_opposite_side_first(self):
        strategy = RuleBasedStrategy(
            rules=[
                {
                    "name": "down_bias_buy",
                    "when": {"metric": "down_price", "operator": "<=", "value": 0.75},
                    "action": "buy_down",
                    "usd_amount": 10,
                }
            ],
        )
        decision = strategy.decide(make_state(up_price=0.25, down_price=0.75, btc_price=98.0, price_to_beat=99.0))
        portfolio = Portfolio(cash=100.0, up_tokens=20.0)

        events = execute_action(
            portfolio=portfolio,
            action=decision.action,
            timestamp="2026-05-30T00:00:00Z",
            up_price=0.25,
            down_price=0.75,
            usd_amount=decision.usd_amount,
            sell_opposite_first=decision.sell_opposite_first,
            reason=decision.reason,
        )

        self.assertEqual(len(events), 2)
        self.assertEqual(events[0].action, "sell")
        self.assertEqual(events[0].side, "up")
        self.assertEqual(events[1].action, "buy")
        self.assertEqual(events[1].side, "down")
        self.assertEqual(portfolio.up_tokens, 0.0)

    def test_max_market_spend_caps_buy_size(self):
        strategy = RuleBasedStrategy(
            rules=[
                {
                    "name": "capped_buy",
                    "when": {"metric": "down_price", "operator": "<=", "value": 0.75},
                    "action": "buy_down",
                    "usd_amount": 10,
                }
            ],
            max_market_spend_usd=12.0,
        )

        decision = strategy.decide(make_state(up_price=0.25, down_price=0.75))
        self.assertEqual(decision.usd_amount, 10.0)

        already_spent_state = make_state(up_price=0.25, down_price=0.75)
        already_spent_state = DecisionState(
            tick=already_spent_state.tick,
            cash=already_spent_state.cash,
            up_tokens=already_spent_state.up_tokens,
            down_tokens=already_spent_state.down_tokens,
            current_balance=already_spent_state.current_balance,
            market_start_balance=already_spent_state.market_start_balance,
            market_spend_used=9.0,
            last_action=already_spent_state.last_action,
            orders_placed=already_spent_state.orders_placed,
        )

        capped_decision = strategy.decide(already_spent_state)
        self.assertEqual(capped_decision.action, "buy_down")
        self.assertEqual(capped_decision.usd_amount, 3.0)

    def test_max_market_spend_holds_after_cap_reached(self):
        strategy = RuleBasedStrategy(
            rules=[
                {
                    "name": "capped_buy",
                    "when": {"metric": "down_price", "operator": "<=", "value": 0.75},
                    "action": "buy_down",
                    "usd_amount": 10,
                }
            ],
            max_market_spend_usd=12.0,
        )
        state = make_state(up_price=0.25, down_price=0.75)
        state = DecisionState(
            tick=state.tick,
            cash=state.cash,
            up_tokens=state.up_tokens,
            down_tokens=state.down_tokens,
            current_balance=state.current_balance,
            market_start_balance=state.market_start_balance,
            market_spend_used=12.0,
            last_action=state.last_action,
            orders_placed=state.orders_placed,
        )

        decision = strategy.decide(state)

        self.assertEqual(decision.action, "hold")
        self.assertEqual(decision.reason, "max market spend reached")

    def test_risk_override_max_price_bypasses_market_spend_cap(self):
        strategy = RuleBasedStrategy(
            rules=[
                {
                    "name": "cheap_down_override",
                    "when": {"metric": "down_price", "operator": "<=", "value": 0.10},
                    "action": "buy_down",
                    "usd_amount": 1,
                    "risk_override_max_price": 0.10,
                }
            ],
            max_market_spend_usd=12.0,
        )
        state = make_state(up_price=0.99, down_price=0.01)
        state = DecisionState(
            tick=state.tick,
            cash=state.cash,
            up_tokens=state.up_tokens,
            down_tokens=state.down_tokens,
            current_balance=state.current_balance,
            market_start_balance=state.market_start_balance,
            market_spend_used=12.0,
            last_action=state.last_action,
            orders_placed=state.orders_placed,
        )

        decision = strategy.decide(state)

        self.assertEqual(decision.action, "buy_down")
        self.assertEqual(decision.usd_amount, 1.0)

    def test_risk_override_max_price_bypasses_max_orders(self):
        strategy = RuleBasedStrategy(
            rules=[
                {
                    "name": "cheap_up_override",
                    "when": {"metric": "up_price", "operator": "<=", "value": 0.10},
                    "action": "buy_up",
                    "usd_amount": 1,
                    "risk_override_max_price": 0.10,
                }
            ],
            max_orders=1,
        )
        state = make_state(up_price=0.01, down_price=0.99)
        state = DecisionState(
            tick=state.tick,
            cash=state.cash,
            up_tokens=state.up_tokens,
            down_tokens=state.down_tokens,
            current_balance=state.current_balance,
            market_start_balance=state.market_start_balance,
            market_spend_used=0.0,
            last_action=state.last_action,
            orders_placed=1,
        )

        decision = strategy.decide(state)

        self.assertEqual(decision.action, "buy_up")
        self.assertEqual(decision.usd_amount, 1.0)

    def test_combine_matching_buys_accumulates_same_side_rules(self):
        strategy = RuleBasedStrategy(
            rules=[
                {
                    "name": "cheap_down",
                    "when": {"metric": "down_price", "operator": "<=", "value": 0.75},
                    "action": "buy_down",
                    "token_amount": 10,
                },
                {
                    "name": "btc_below_down_chaser",
                    "when": {"metric": "btc_below_price_to_beat", "operator": "is", "value": True},
                    "action": "buy_down",
                    "token_amount": 20,
                },
                {
                    "name": "up_chaser_not_combined",
                    "when": {"metric": "up_price", "operator": "<=", "value": 0.25},
                    "action": "buy_up",
                    "token_amount": 50,
                },
            ],
            combine_matching_buys=True,
        )

        decision = strategy.decide(make_state(
            up_price=0.25,
            down_price=0.75,
            btc_price=98.0,
            price_to_beat=99.0,
        ))

        self.assertEqual(decision.action, "buy_down")
        self.assertEqual(decision.usd_amount, 22.5)
        self.assertEqual(decision.reason, "cheap_down + btc_below_down_chaser")

    def test_first_match_wins_without_combine_matching_buys(self):
        strategy = RuleBasedStrategy(
            rules=[
                {
                    "name": "cheap_down",
                    "when": {"metric": "down_price", "operator": "<=", "value": 0.75},
                    "action": "buy_down",
                    "token_amount": 10,
                },
                {
                    "name": "btc_below_down_chaser",
                    "when": {"metric": "btc_below_price_to_beat", "operator": "is", "value": True},
                    "action": "buy_down",
                    "token_amount": 20,
                },
            ],
        )

        decision = strategy.decide(make_state(up_price=0.25, down_price=0.75))

        self.assertEqual(decision.action, "buy_down")
        self.assertEqual(decision.usd_amount, 7.5)
        self.assertEqual(decision.reason, "cheap_down")


if __name__ == "__main__":
    unittest.main()
