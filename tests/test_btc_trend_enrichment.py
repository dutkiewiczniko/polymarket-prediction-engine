import csv
import tempfile
import unittest
from pathlib import Path

from scripts.data.enrich_markets_with_btc_trend import (
    BtcPriceHistory,
    compute_market_trends,
    enrich_market_csv,
    parse_periods,
    trend_column_name,
)
from simulator.models import DecisionState
from simulator.replay import load_market_ticks
from simulator.strategies import RuleBasedStrategy


START_TS = 1778894100  # aligned to the 5m kline grid


def write_market_csv(path: Path, btc_first_row: str = "80000") -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "unix_time", "seconds_left", "elapsed",
            "up_price", "down_price", "btc_binance", "btc_chainlink", "price_to_beat",
        ])
        writer.writerow(["00:00:00.5", START_TS + 0.5, 299.5, 0.5, 0.5, 0.5, btc_first_row, "", ""])
        writer.writerow(["00:00:01.0", START_TS + 1.0, 299.0, 1.0, 0.5, 0.5, "80001", "", "80000"])


def make_history(prices: dict[int, float]) -> BtcPriceHistory:
    history = BtcPriceHistory(Path(tempfile.mkdtemp()) / "cache.csv")
    history.prices = dict(prices)
    return history


class ParsePeriodsTest(unittest.TestCase):
    def test_parses_default_style_spec(self):
        periods = parse_periods("1h,12h,24h,7d")
        self.assertEqual(periods, {"1h": 3600, "12h": 43200, "24h": 86400, "7d": 604800})

    def test_rejects_unknown_unit(self):
        with self.assertRaises(ValueError):
            parse_periods("5x")


class ComputeMarketTrendsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.market_csv = self.tmp / f"btc-updown-5m-{START_TS}.csv"

    def test_change_pct_from_csv_baseline(self):
        write_market_csv(self.market_csv, btc_first_row="80000")
        history = make_history({
            START_TS: 80080.0,  # within 0.5% of csv baseline -> csv baseline wins
            START_TS - 3600: 79200.0,
        })
        trend = compute_market_trends(self.market_csv, {"1h": 3600}, history)
        self.assertEqual(trend["baseline_source"], "market_csv")
        self.assertAlmostEqual(trend["values"]["1h"], (80000 - 79200) / 79200 * 100.0)

    def test_invalid_csv_baseline_falls_back_to_kline(self):
        write_market_csv(self.market_csv, btc_first_row="123")  # glitched feed value
        history = make_history({
            START_TS: 80000.0,
            START_TS - 3600: 80000.0,
        })
        trend = compute_market_trends(self.market_csv, {"1h": 3600}, history)
        self.assertEqual(trend["baseline_source"], "binance_kline_csv_mismatch")
        self.assertAlmostEqual(trend["values"]["1h"], 0.0)

    def test_missing_history_gives_none(self):
        write_market_csv(self.market_csv)
        history = make_history({START_TS: 80000.0})
        trend = compute_market_trends(self.market_csv, {"1h": 3600}, history)
        self.assertIsNone(trend["values"]["1h"])


class EnrichMarketCsvTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.market_csv = self.tmp / f"btc-updown-5m-{START_TS}.csv"
        write_market_csv(self.market_csv)
        self.periods = {"1h": 3600}
        self.history = make_history({
            START_TS: 80000.0,
            START_TS - 3600: 79200.0,
        })

    def _trend(self):
        return compute_market_trends(self.market_csv, self.periods, self.history)

    def test_appends_constant_column_to_every_row(self):
        status = enrich_market_csv(self.market_csv, self._trend(), self.periods, force=False)
        self.assertEqual(status, "enriched")
        with self.market_csv.open(newline="") as f:
            rows = list(csv.DictReader(f))
        expected = f"{(80000 - 79200) / 79200 * 100.0:.6f}"
        self.assertTrue(all(row[trend_column_name("1h")] == expected for row in rows))

    def test_second_run_is_idempotent(self):
        enrich_market_csv(self.market_csv, self._trend(), self.periods, force=False)
        status = enrich_market_csv(self.market_csv, self._trend(), self.periods, force=False)
        self.assertEqual(status, "already_enriched")
        with self.market_csv.open(newline="") as f:
            header = f.readline().strip().split(",")
        self.assertEqual(header.count(trend_column_name("1h")), 1)

    def test_force_overwrites_in_place_without_duplicating(self):
        enrich_market_csv(self.market_csv, self._trend(), self.periods, force=False)
        self.history.prices[START_TS - 3600] = 80000.0  # changed history -> changed value
        status = enrich_market_csv(self.market_csv, self._trend(), self.periods, force=True)
        self.assertEqual(status, "enriched")
        with self.market_csv.open(newline="") as f:
            reader = csv.DictReader(f)
            self.assertEqual(reader.fieldnames.count(trend_column_name("1h")), 1)
            rows = list(reader)
        self.assertTrue(all(row[trend_column_name("1h")] == "0.000000" for row in rows))


class TrendColumnsAsRuleMetricsTest(unittest.TestCase):
    """Enriched columns must flow through tick.extra into rule metrics."""

    def test_rule_fires_on_trend_column(self):
        tmp = Path(tempfile.mkdtemp())
        market_csv = tmp / f"btc-updown-5m-{START_TS}.csv"
        write_market_csv(market_csv)
        periods = {"24h": 86400}
        history = make_history({
            START_TS: 80000.0,
            START_TS - 86400: 78400.0,  # +2.04% into market start
        })
        trend = compute_market_trends(market_csv, periods, history)
        enrich_market_csv(market_csv, trend, periods, force=False)

        ticks = load_market_ticks(market_csv)
        self.assertIn(trend_column_name("24h"), ticks[0].extra)

        strategy = RuleBasedStrategy(rules=[
            {
                "name": "strong_24h_uptrend_buy_up",
                "when": {"metric": "btc_change_24h_pct", "operator": ">=", "value": 1.0},
                "action": "buy_up",
                "usd_amount": 2,
            },
        ])
        state = DecisionState(
            tick=ticks[0],
            cash=100.0,
            up_tokens=0.0,
            down_tokens=0.0,
            current_balance=100.0,
            market_start_balance=100.0,
            market_spend_used=0.0,
            last_action="none",
            orders_placed=0,
        )
        decision = strategy.decide(state)
        self.assertEqual(decision.action, "buy_up")
        self.assertEqual(decision.reason, "strong_24h_uptrend_buy_up")


if __name__ == "__main__":
    unittest.main()
