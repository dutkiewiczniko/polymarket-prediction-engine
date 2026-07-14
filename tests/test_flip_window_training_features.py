import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.training.train_flip_window_model import (
    FeatureConfig,
    build_feature_frame,
    leakage_sentinel,
    load_manifest,
)


class FlipWindowFeatureTests(unittest.TestCase):
    def test_entry_features_exclude_future_length_and_absolute_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            dataset_dir = Path(tmp)
            windows_dir = dataset_dir / "windows"
            windows_dir.mkdir()

            (dataset_dir / "manifest.csv").write_text(
                "\n".join(
                    [
                        "window_file,market,side,cohort,episode_num,entry_price,entry_seconds_left,entry_unix_time,entry_deficit_pct,crash_speed_30s,hour_utc,n_ticks,window_end_reason,won_resolution,max_price_after,time_to_50_s,recovered_50,recovered_75,recovered_90,recovered_95,has_hist_300s,has_hist_600s,has_hist_900s",
                        "w.csv,m1,up,may_bench,1,0.19,120,1000,0.03,0.2,9,500,market_end,False,0.2,,False,False,False,False,True,False,False",
                    ]
                ),
                encoding="utf-8",
            )
            (windows_dir / "w.csv").write_text(
                "\n".join(
                    [
                        "unix_time,seconds_left,time_in_window_s,price,deficit_pct,best_bid",
                        "1000,120,0,0.19,0.03,0.18",
                        "1001,119,1,0.20,0.04,0.19",
                    ]
                ),
                encoding="utf-8",
            )

            manifest = load_manifest(dataset_dir, "won_resolution")
            features, _, feature_columns = build_feature_frame(
                manifest,
                dataset_dir,
                FeatureConfig(
                    framing="entry",
                    early_seconds=30,
                    target="won_resolution",
                    train_cohort="may_bench",
                    holdout_cohort="july_live",
                ),
            )

        self.assertNotIn("manifest__n_ticks", feature_columns)
        self.assertNotIn("tick_entry__unix_time", feature_columns)
        self.assertIn("tick_entry__price", feature_columns)
        self.assertEqual(float(features.loc[0, "tick_entry__price"]), 0.19)


class LeakageSentinelTests(unittest.TestCase):
    def test_flags_label_encoding_feature_and_spares_noise(self):
        rng = np.random.default_rng(0)
        y = np.array([0, 1] * 25)
        x = pd.DataFrame(
            {
                # Near-perfect encoder of the label -> must be flagged.
                "leaky": y + rng.normal(0, 0.01, size=len(y)),
                # Pure noise -> must not be flagged.
                "noise": rng.normal(0, 1, size=len(y)),
            }
        )
        report = {r["feature"]: r for r in leakage_sentinel(x, y, warn_auc=0.80)}
        self.assertTrue(report["leaky"]["suspicious"])
        self.assertFalse(report["noise"]["suspicious"])
        self.assertGreater(report["leaky"]["separation"], report["noise"]["separation"])


if __name__ == "__main__":
    unittest.main()
