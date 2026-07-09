import unittest

from simulator.models import DecisionState, MarketTick
from simulator.strategies import (
    RuleBasedStrategy,
    latest_value_at_or_before,
    momentum_pct,
    momentum_pct_by_ticks,
)


MISSING = object()


def make_tick(*, unix_time, up_price=0.5, down_price=0.5, btc_price=100.0):
    btc_binance = None if btc_price is MISSING else btc_price
    return MarketTick(
        timestamp="2026-07-07T00:00:00Z",
        unix_time=unix_time,
        seconds_left=120.0,
        elapsed=10.0,
        up_price=up_price,
        down_price=down_price,
        btc_binance=btc_binance,
        btc_chainlink=None,
        price_to_beat=99.0,
    )


def make_state(tick):
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


class MomentumHelperTests(unittest.TestCase):
    def test_latest_value_at_or_before_finds_most_recent_match(self):
        series = [(0.0, 100.0), (1.0, 101.0), (2.0, 102.0), (3.0, 103.0)]
        self.assertEqual(latest_value_at_or_before(series, 2.5), 102.0)
        self.assertEqual(latest_value_at_or_before(series, 3.0), 103.0)
        self.assertIsNone(latest_value_at_or_before(series, -1.0))

    def test_momentum_pct_computes_pct_change_over_window(self):
        series = [(0.0, 100.0), (10.0, 110.0)]
        # current=110 at t=10, looking back 10s lands on the t=0 sample (100).
        self.assertAlmostEqual(momentum_pct(series, 10.0, 10.0), 10.0)

    def test_momentum_pct_none_without_enough_history(self):
        series = [(10.0, 110.0)]
        self.assertIsNone(momentum_pct(series, 10.0, 60.0))

    def test_momentum_pct_none_for_empty_series(self):
        self.assertIsNone(momentum_pct([], 10.0, 1.0))


class MomentumTickHelperTests(unittest.TestCase):
    def test_momentum_pct_by_ticks_looks_back_exact_tick_count(self):
        series = [(0.0, 100.0), (0.2, 101.0), (0.4, 102.0), (0.6, 103.0)]
        self.assertAlmostEqual(momentum_pct_by_ticks(series, 1), pct_change_expected(103.0, 102.0))
        self.assertAlmostEqual(momentum_pct_by_ticks(series, 3), pct_change_expected(103.0, 100.0))

    def test_momentum_pct_by_ticks_none_without_enough_ticks(self):
        series = [(0.0, 100.0), (0.2, 101.0)]
        self.assertIsNone(momentum_pct_by_ticks(series, 5))

    def test_momentum_pct_by_ticks_none_for_empty_series(self):
        self.assertIsNone(momentum_pct_by_ticks([], 1))


def pct_change_expected(current, previous):
    return ((current - previous) / previous) * 100.0


class RuleBasedStrategyTickMomentumMetricTests(unittest.TestCase):
    def test_tick_momentum_reacts_faster_than_time_windows(self):
        strategy = RuleBasedStrategy(rules=[])
        # Three ticks packed within 0.4s (sub-second spacing, like real data).
        strategy.decide(make_state(make_tick(unix_time=0.0, btc_price=100.0)))
        strategy.decide(make_state(make_tick(unix_time=0.2, btc_price=100.0)))
        state = make_state(make_tick(unix_time=0.4, btc_price=101.0))
        metrics = strategy._build_metrics(state)

        # momentum_1t compares to the immediately preceding tick (100.0 -> 101.0).
        self.assertAlmostEqual(metrics["momentum_1t"], 1.0)
        # momentum_2t compares to two ticks back (also 100.0 here) -> same magnitude.
        self.assertAlmostEqual(metrics["momentum_2t"], 1.0)
        # momentum_1s requires a sample from >=1s ago; none exists yet.
        self.assertIsNone(metrics["momentum_1s"])

    def test_tick_momentum_metrics_present_for_all_tick_windows(self):
        strategy = RuleBasedStrategy(rules=[])
        state = make_state(make_tick(unix_time=0.0))
        metrics = strategy._build_metrics(state)
        for ticks in RuleBasedStrategy.MOMENTUM_TICK_WINDOWS:
            self.assertIn(f"momentum_{ticks}t", metrics)
            self.assertIn(f"up_price_momentum_{ticks}t", metrics)
            self.assertIn(f"down_price_momentum_{ticks}t", metrics)


class RuleBasedStrategyMomentumMetricTests(unittest.TestCase):
    def test_momentum_metrics_use_elapsed_time_not_tick_count(self):
        strategy = RuleBasedStrategy(rules=[])

        # Ten ticks packed into the first second (sub-second spacing, like real data),
        # then one tick 30s later. A tick-count lookback (e.g. "10 ticks ago") would
        # land on the very first sample; a correct time-based lookback uses whatever
        # price was known 30s ago, which is also the first sample here -- either way
        # the window math itself (not just "which tick") must produce the right pct.
        for i in range(10):
            strategy.decide(make_state(make_tick(unix_time=i * 0.1, btc_price=100.0)))

        state = make_state(make_tick(unix_time=30.0, btc_price=110.0))
        metrics = strategy._build_metrics(state)

        self.assertAlmostEqual(metrics["momentum_30s"], 10.0)

    def test_momentum_metric_is_none_when_no_sample_predates_the_window(self):
        strategy = RuleBasedStrategy(rules=[])
        # First tick ever seen -- there is no price from "120s ago" to compare against.
        state = make_state(make_tick(unix_time=1000.0, btc_price=100.0))
        metrics = strategy._build_metrics(state)
        self.assertIsNone(metrics["momentum_120s"])

    def test_momentum_metrics_present_for_all_configured_windows(self):
        strategy = RuleBasedStrategy(rules=[])
        state = make_state(make_tick(unix_time=0.0))
        metrics = strategy._build_metrics(state)
        for window_s in RuleBasedStrategy.MOMENTUM_WINDOWS_S:
            label = f"momentum_{int(window_s)}s"
            self.assertIn(label, metrics)
            self.assertIn(f"up_price_momentum_{int(window_s)}s", metrics)
            self.assertIn(f"down_price_momentum_{int(window_s)}s", metrics)

    def test_seed_btc_history_gives_long_windows_data_at_market_start(self):
        strategy = RuleBasedStrategy(rules=[])
        # Prior market: BTC at 100.0 from t=0..300 (samples every 5s).
        strategy.seed_btc_history([(float(t), 100.0) for t in range(0, 300, 5)])

        # First tick of the new market at t=300, BTC now 110.
        state = make_state(make_tick(unix_time=300.0, btc_price=110.0))
        metrics = strategy._build_metrics(state)

        # 120s window reaches back into the seeded prior-market history.
        self.assertAlmostEqual(metrics["momentum_120s"], 10.0)

    def test_seed_btc_history_does_not_emit_btc_momentum_without_current_btc(self):
        strategy = RuleBasedStrategy(rules=[])
        strategy.seed_btc_history([(float(t), 100.0 + t) for t in range(0, 300, 5)])

        state = make_state(make_tick(unix_time=300.0, btc_price=MISSING))
        metrics = strategy._build_metrics(state)

        self.assertIsNone(metrics["momentum_120s"])
        self.assertIsNone(metrics["momentum_1t"])

    def test_seed_btc_history_does_not_touch_token_price_series(self):
        strategy = RuleBasedStrategy(rules=[])
        strategy.seed_btc_history([(float(t), 100.0) for t in range(0, 300, 5)])

        state = make_state(make_tick(unix_time=300.0, btc_price=110.0))
        metrics = strategy._build_metrics(state)

        # Token prices reset each market; their momentum must stay None on tick 1.
        self.assertIsNone(metrics["up_price_momentum_120s"])
        self.assertIsNone(metrics["down_price_momentum_120s"])

    def test_old_samples_are_trimmed_from_series(self):
        strategy = RuleBasedStrategy(rules=[])
        buffer_s = RuleBasedStrategy._MOMENTUM_HISTORY_BUFFER_S
        horizon = buffer_s * 2
        step = 5.0
        last_t = 0.0
        t = 0.0
        while t < horizon:
            strategy.decide(make_state(make_tick(unix_time=t, btc_price=100.0 + t)))
            last_t = t
            t += step

        # The series should never hold samples older than the history buffer
        # relative to the latest tick (one tick of slack allowed).
        oldest_kept = strategy._btc_series[0][0]
        self.assertGreaterEqual(oldest_kept, last_t - buffer_s - step)


if __name__ == "__main__":
    unittest.main()
