import unittest

from simulator.batch import DEFAULT_EFFECTIVE_MARKET_BALANCE_BANDS, resolve_effective_market_balance


class DefaultEffectiveMarketBalanceTests(unittest.TestCase):
    def test_empty_config_falls_back_to_default_bands_instead_of_none(self):
        result = resolve_effective_market_balance(400.0, {})
        self.assertIsNotNone(result)
        self.assertEqual(result, 60.0)  # [150, 750, 60) band

    def test_default_bands_match_documented_prop_tighter_after60_curve(self):
        cases = {
            10.0: 5.0,
            20.0: 10.0,
            50.0: 15.0,
            100.0: 30.0,
            500.0: 60.0,
            1000.0: 120.0,
            20000.0: 1920.0,
        }
        for balance, expected in cases.items():
            with self.subTest(balance=balance):
                self.assertEqual(resolve_effective_market_balance(balance, {}), expected)

    def test_explicit_bands_still_override_the_default(self):
        custom_cfg = {"effective_market_balance_bands": [[1, None, 200]]}
        result = resolve_effective_market_balance(400.0, custom_cfg)
        self.assertEqual(result, 200.0)

    def test_explicit_pct_still_overrides_the_default(self):
        custom_cfg = {"effective_market_balance_pct": 0.5}
        result = resolve_effective_market_balance(400.0, custom_cfg)
        self.assertEqual(result, 200.0)

    def test_default_bands_constant_is_sorted_and_covers_from_one(self):
        self.assertEqual(DEFAULT_EFFECTIVE_MARKET_BALANCE_BANDS[0][0], 1)
        self.assertIsNone(DEFAULT_EFFECTIVE_MARKET_BALANCE_BANDS[-1][1])


if __name__ == "__main__":
    unittest.main()
