import unittest

from scripts.live.live_strategy_suite import (
    BTC_WARMUP_RETENTION_S,
    collect_btc_warmup_samples,
)
from simulator.config_loader import load_strategy_from_yaml
from simulator.models import DecisionState, MarketTick
from simulator.strategies import RuleBasedStrategy


MARKET_START = 1_779_000_000.0


def make_state(unix_time, btc_price=80000.0):
    tick = MarketTick(
        timestamp="00:00:00.5",
        unix_time=unix_time,
        seconds_left=299.5,
        elapsed=0.5,
        up_price=0.5,
        down_price=0.5,
        btc_binance=btc_price,
        btc_chainlink=None,
        price_to_beat=btc_price,
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


class CollectBtcWarmupSamplesTest(unittest.TestCase):
    def test_binance_only(self):
        binance = [(1.0, 100.0), (2.0, 101.0)]
        self.assertEqual(collect_btc_warmup_samples(binance, []), binance)

    def test_chainlink_preferred_with_binance_backfill(self):
        binance = [(1.0, 100.0), (2.0, 101.0), (3.0, 102.0)]
        chainlink = [(2.5, 200.0), (3.5, 201.0)]
        merged = collect_btc_warmup_samples(binance, chainlink)
        # binance strictly before first chainlink sample, then chainlink only
        self.assertEqual(merged, [(1.0, 100.0), (2.0, 101.0), (2.5, 200.0), (3.5, 201.0)])
        # timestamps stay sorted (seed_btc_history requires ascending series)
        self.assertEqual([ts for ts, _ in merged], sorted(ts for ts, _ in merged))

    def test_retention_covers_longest_engine_window(self):
        self.assertGreaterEqual(BTC_WARMUP_RETENTION_S, max(RuleBasedStrategy.MOMENTUM_WINDOWS_S))


class SeededStrategyHasLongWindowsAtOpenTest(unittest.TestCase):
    """The point of the warmup port: slow_plus_1t's 900s window must be alive on
    the very first tick of a live market when seeded with prior history."""

    def _seeded_metrics(self, seed: bool):
        strategy, _ = load_strategy_from_yaml(
            "configs/strategies/momentum_metrics_experiment/combos/combo_slow_plus_1t.yaml"
        )
        if seed:
            samples = [
                (MARKET_START - offset_s, 79000.0 + offset_s)
                for offset_s in range(int(BTC_WARMUP_RETENTION_S), 0, -5)
            ]
            strategy.seed_btc_history(collect_btc_warmup_samples(samples, []))
        state = make_state(MARKET_START + 0.5)
        return strategy._build_metrics(state)

    def test_cold_start_long_windows_are_none(self):
        metrics = self._seeded_metrics(seed=False)
        self.assertIsNone(metrics["momentum_900s"])
        self.assertIsNone(metrics["momentum_240s"])

    def test_seeded_long_windows_are_live_on_first_tick(self):
        metrics = self._seeded_metrics(seed=True)
        self.assertIsNotNone(metrics["momentum_900s"])
        self.assertIsNotNone(metrics["momentum_240s"])
        self.assertIsNotNone(metrics["momentum_120s"])
        self.assertIsNotNone(metrics["momentum_60s"])


if __name__ == "__main__":
    unittest.main()
