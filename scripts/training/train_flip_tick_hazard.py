"""Per-tick hazard model: P(down-side recovers to >=0.50 from HERE).

Reframes flip prediction from window-entry classification (near-zero signal --
see train_flip_window_model.py results) to per-tick decision points: every
sampled tick of a flip_windows_v1 window is a row whose features are that
tick's trailing state and whose label is whether the price later crosses the
exit (0.50) before the market ends. The depth/trajectory information that
separated FLIP from DEAD windows becomes model input instead of lost context.

ADAPTIVE SAMPLING (so fast moves are densely represented and flat stretches
are not): a tick is sampled when either
  * --base-interval-s (default 5s) has passed since the last sample, or
  * the price moved >= --move-threshold (default 0.02) since the last sample
    and --fast-interval-s (default 1s) has passed.

Leakage rules learned from the window-level harness:
  * the recovery-crossing tick (price >= exit) is NEVER a sample -- it IS the
    label; all other ticks are legitimate decision points (their label is
    strictly about the future).
  * samples require seconds_left >= --min-seconds-left (no untradeable rows).
  * leakage sentinel + prior/logistic baselines reused from the window harness.

EVALUATION IS EDGE-BASED, NOT AUC: buying $1 at price p and selling at 0.50
pays 0.5/p - 1 on recovery, -1 otherwise => breakeven probability is 2p. The
trading policy is "buy at the first sampled tick where P_hat > 2p + margin"
(one trade per window). Reported on May out-of-fold probs and on the July
holdout, next to a buy-every-sample baseline. ROC etc. are still printed, but
the policy P&L is the number that matters.

Usage:
    python scripts/training/train_flip_tick_hazard.py
    python scripts/training/train_flip_tick_hazard.py --target won_resolution
"""

import argparse
import importlib.util
import sys
from dataclasses import asdict
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_spec = importlib.util.spec_from_file_location(
    "flip_window_harness", _HERE / "train_flip_window_model.py")
_harness = importlib.util.module_from_spec(_spec)
sys.modules["flip_window_harness"] = _harness
_spec.loader.exec_module(_harness)

bool_series = _harness.bool_series
leakage_sentinel = _harness.leakage_sentinel
evaluate_predictions = _harness.evaluate_predictions
threshold_for_recall = _harness.threshold_for_recall
grouped_split_indices = _harness.grouped_split_indices
grouped_cv = _harness.grouped_cv
prior_baseline_factory = _harness.prior_baseline_factory
logistic_baseline_factory = _harness.logistic_baseline_factory
make_model = _harness.make_model
hgb_factory = _harness.hgb_factory
write_json = _harness.write_json
print_metrics = _harness.print_metrics

DEFAULT_DATASET_DIR = Path("flip_windows_v1")
DEFAULT_OUTPUT_DIR = Path("models/flip_tick_hazard_v1")
EXIT_PRICE = 0.50
MANIFEST_FEATURES = ["entry_price", "entry_seconds_left", "crash_speed_30s",
                     "episode_num", "has_hist_600s", "has_hist_900s"]
EXCLUDED_TICK_COLUMNS = {"unix_time"}


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Per-tick hazard model with edge-based policy evaluation.")
    ap.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    ap.add_argument("--target", default="recovered_50",
                    choices=["recovered_50", "won_resolution"])
    ap.add_argument("--base-interval-s", type=float, default=5.0,
                    help="regular sampling interval. Default 5s.")
    ap.add_argument("--fast-interval-s", type=float, default=1.0,
                    help="min gap when the movement trigger fires. Default 1s.")
    ap.add_argument("--move-threshold", type=float, default=0.02,
                    help="price move since last sample that triggers fast "
                         "sampling. Default 0.02.")
    ap.add_argument("--min-seconds-left", type=float, default=5.0)
    ap.add_argument("--edge-margin", type=float, default=0.05,
                    help="required P_hat - breakeven margin before the policy "
                         "buys. Default 0.05.")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-iter", type=int, default=300)
    ap.add_argument("--learning-rate", type=float, default=0.04)
    ap.add_argument("--max-leaf-nodes", type=int, default=31)
    ap.add_argument("--l2-regularization", type=float, default=0.1)
    ap.add_argument("--min-samples-leaf", type=int, default=60)
    ap.add_argument("--train-cohort", default="may_bench")
    ap.add_argument("--holdout-cohort", default="july_live")
    ap.add_argument("--leakage-warn-auc", type=float, default=0.80)
    ap.add_argument("--skip-baselines", action="store_true")
    return ap.parse_args()


def sample_window(df: pd.DataFrame, args) -> np.ndarray:
    """Adaptive-sampling row indices for one window CSV."""
    times = df["unix_time"].to_numpy(dtype=float)
    prices = df["price"].to_numpy(dtype=float)
    seconds_left = df["seconds_left"].to_numpy(dtype=float)
    take = []
    last_t, last_p = -np.inf, np.nan
    for i in range(len(df)):
        if seconds_left[i] < args.min_seconds_left:
            continue
        p = prices[i]
        if not np.isfinite(p) or p >= EXIT_PRICE:  # crossing tick = the label
            continue
        dt = times[i] - last_t
        moved = np.isfinite(last_p) and abs(p - last_p) >= args.move_threshold
        if dt >= args.base_interval_s or (moved and dt >= args.fast_interval_s):
            take.append(i)
            last_t, last_p = times[i], p
    return np.asarray(take, dtype=int)


def build_tick_dataset(manifest: pd.DataFrame, dataset_dir: Path,
                       args) -> tuple[pd.DataFrame, pd.DataFrame]:
    """One row per sampled tick: features + metadata (label, market, price)."""
    windows_dir = dataset_dir / "windows"
    feat_rows, meta_rows = [], []
    for rec in manifest.to_dict("records"):
        path = windows_dir / str(rec["window_file"])
        df = pd.read_csv(path)
        idx = sample_window(df, args)
        if idx.size == 0:
            continue
        label = bool(rec[args.target])
        tick_cols = [c for c in df.columns if c not in EXCLUDED_TICK_COLUMNS]
        sub = df.iloc[idx]
        base = {f"manifest__{c}": float(rec[c]) for c in MANIFEST_FEATURES}
        base["manifest__side_is_up"] = 1.0 if rec["side"] == "up" else 0.0
        for _, row in sub.iterrows():
            feats = dict(base)
            for c in tick_cols:
                feats[f"tick__{c}"] = row[c]
            feat_rows.append(feats)
            meta_rows.append({
                "window_file": rec["window_file"], "market": rec["market"],
                "cohort": rec["cohort"], "side": rec["side"],
                "price": float(row["price"]),
                "seconds_left": float(row["seconds_left"]),
                "time_in_window_s": float(row["time_in_window_s"]),
                "unix_time": float(row["unix_time"]),
                "y": int(label),
            })
    x = pd.DataFrame(feat_rows).replace([np.inf, -np.inf], np.nan)
    meta = pd.DataFrame(meta_rows)
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")
    return x, meta


def policy_pnl(meta: pd.DataFrame, prob: np.ndarray, args,
               label: str) -> dict:
    """One $1 buy per window at the first sampled tick with edge > margin.

    Payoff: recovered_50 target = sell at the 0.50 exit (0.5/p - 1 or -1);
    won_resolution target = hold to resolution (1/p - 1 or -1).
    """
    m = meta.copy()
    m["prob"] = prob
    m["breakeven"] = (2.0 * m["price"] if args.target == "recovered_50"
                      else m["price"])
    m["edge"] = m["prob"] - m["breakeven"]
    trades = []
    for wf, g in m.groupby("window_file", sort=False):
        g = g.sort_values("unix_time")
        hit = g[g["edge"] > args.edge_margin]
        if hit.empty:
            continue
        t = hit.iloc[0]
        p = t["price"]
        won = bool(t["y"])
        payoff = ((EXIT_PRICE / p - 1.0) if args.target == "recovered_50"
                  else (1.0 / p - 1.0)) if won else -1.0
        trades.append({"window_file": wf, "price": p, "prob": t["prob"],
                       "edge": t["edge"], "won": won, "payoff": payoff,
                       "seconds_left": t["seconds_left"]})
    tdf = pd.DataFrame(trades)
    n_windows = m["window_file"].nunique()
    # buy-every-sample baseline (what an indiscriminate buyer would make);
    # clip price away from 0 so a 0.001 print cannot produce infinite payoff
    p_all = np.maximum(m["price"].to_numpy(), 0.005)
    every = np.where(
        m["y"].astype(bool),
        (EXIT_PRICE / p_all - 1.0) if args.target == "recovered_50"
        else (1.0 / p_all - 1.0),
        -1.0)
    out = {
        "label": label, "windows": int(n_windows), "trades": int(len(tdf)),
        "buy_every_sample_ev": float(np.mean(every)),
        "edge_margin": args.edge_margin,
    }
    if len(tdf):
        out.update({
            "wins": int(tdf["won"].sum()),
            "hit_rate": float(tdf["won"].mean()),
            "ev_per_dollar": float(tdf["payoff"].mean()),
            "total_pnl": float(tdf["payoff"].sum()),
            "median_entry_price": float(tdf["price"].median()),
            "median_seconds_left": float(tdf["seconds_left"].median()),
        })
    return out, tdf


def print_policy(res: dict) -> None:
    print(f"  {res['label']}: windows={res['windows']} trades={res['trades']} "
          f"(buy-every-sample EV {res['buy_every_sample_ev']:+.3f})")
    if res.get("trades"):
        print(f"    wins={res['wins']} ({100 * res['hit_rate']:.1f}%) "
              f"EV/$1={res['ev_per_dollar']:+.3f} total={res['total_pnl']:+.1f} "
              f"median entry p={res['median_entry_price']:.2f} "
              f"sec_left={res['median_seconds_left']:.0f}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    manifest = pd.read_csv(args.dataset_dir / "manifest.csv")
    manifest[args.target] = bool_series(manifest[args.target])
    for c in ("has_hist_600s", "has_hist_900s"):
        manifest[c] = bool_series(manifest[c]).astype(float)

    print("building tick dataset (adaptive sampling: "
          f"{args.base_interval_s}s base, {args.fast_interval_s}s on moves "
          f">= {args.move_threshold})...")
    x, meta = build_tick_dataset(manifest, args.dataset_dir, args)
    y = meta["y"].to_numpy()
    print(f"samples: {len(x)} from {meta['window_file'].nunique()} windows / "
          f"{meta['market'].nunique()} markets; positive rate {y.mean():.2%}")
    for cohort, g in meta.groupby("cohort"):
        print(f"  {cohort}: {len(g)} samples, {g['y'].mean():.2%} positive")

    train_mask = meta["cohort"].eq(args.train_cohort).to_numpy()
    holdout_mask = meta["cohort"].eq(args.holdout_cohort).to_numpy()
    x_train = x.loc[train_mask].reset_index(drop=True)
    y_train = y[train_mask]
    meta_train = meta.loc[train_mask].reset_index(drop=True)
    groups = meta_train["market"].astype(str).to_numpy()
    x_hold = x.loc[holdout_mask].reset_index(drop=True)
    y_hold = y[holdout_mask]
    meta_hold = meta.loc[holdout_mask].reset_index(drop=True)

    sentinel = leakage_sentinel(x_train, y_train, args.leakage_warn_auc)
    flagged = [s["feature"] for s in sentinel if s["suspicious"]]
    if flagged:
        print(f"\nLEAKAGE WARNING: {len(flagged)} feature(s) over AUC "
              f"{args.leakage_warn_auc}:")
        for s in sentinel[:8]:
            if s["suspicious"]:
                print(f"  {s['feature']:<32} AUC={s['auc']:.3f}")
    top = sentinel[0]
    print(f"strongest single feature: {top['feature']} AUC={top['auc']:.3f}")

    split_indices = grouped_split_indices(x_train, y_train, groups, args)
    oof, folds, threshold = grouped_cv(
        x_train, y_train, groups, args, hgb_factory(args), split_indices)
    oof_prob = oof["y_prob"].to_numpy()
    oof_metrics = evaluate_predictions(y_train, oof_prob, threshold)

    model = make_model(args, positive_rate=float(np.mean(y_train)))
    model.fit(x_train, y_train)
    hold_prob = model.predict_proba(x_hold)[:, 1]
    hold_metrics = evaluate_predictions(y_hold, hold_prob, threshold)

    baselines = {}
    if not args.skip_baselines:
        for name, factory in (("prior", prior_baseline_factory()),
                              ("logistic", logistic_baseline_factory(args))):
            b_oof, _, b_thr = grouped_cv(x_train, y_train, groups, args,
                                         factory, split_indices)
            b_model = factory(float(np.mean(y_train)))
            b_model.fit(x_train, y_train)
            b_hold = b_model.predict_proba(x_hold)[:, 1]
            baselines[name] = {
                "out_of_fold": evaluate_predictions(
                    y_train, b_oof["y_prob"].to_numpy(), b_thr),
                "holdout": evaluate_predictions(y_hold, b_hold, b_thr),
            }
            if name == "logistic":
                logit_hold_prob = b_hold
                logit_oof_prob = b_oof["y_prob"].to_numpy()

    print("\nEvaluation (classification view)")
    print("----------")
    if baselines:
        print_metrics("Prior", baselines["prior"]["out_of_fold"])
        print_metrics("Logit OOF", baselines["logistic"]["out_of_fold"])
        print_metrics("Logit July", baselines["logistic"]["holdout"])
    print_metrics("Tree OOF", oof_metrics)
    print_metrics("Tree July", hold_metrics)

    print("\nPolicy P&L (buy $1 when P_hat > breakeven + "
          f"{args.edge_margin}; one trade per window)")
    print("----------")
    pol_oof, trades_oof = policy_pnl(meta_train, oof_prob, args,
                                     "Tree May-OOF")
    pol_hold, trades_hold = policy_pnl(meta_hold, hold_prob, args,
                                       "Tree July")
    print_policy(pol_oof)
    print_policy(pol_hold)
    policy_results = {"tree_may_oof": pol_oof, "tree_july": pol_hold}
    if baselines:
        pol_l_oof, _ = policy_pnl(meta_train, logit_oof_prob, args,
                                  "Logit May-OOF")
        pol_l_hold, _ = policy_pnl(meta_hold, logit_hold_prob, args,
                                   "Logit July")
        print_policy(pol_l_oof)
        print_policy(pol_l_hold)
        policy_results["logit_may_oof"] = pol_l_oof
        policy_results["logit_july"] = pol_l_hold

    metrics = {
        "config": {**{k: (str(v) if isinstance(v, Path) else v)
                      for k, v in vars(args).items()}},
        "samples": {"total": int(len(x)),
                    "by_cohort": {c: int(n) for c, n
                                  in meta.groupby("cohort").size().items()}},
        "grouped_cv": {"folds": folds, "out_of_fold": oof_metrics},
        "holdout": hold_metrics,
        "baselines": baselines,
        "policy": policy_results,
        "leakage_sentinel": sentinel[:25],
    }
    write_json(args.output_dir / "metrics.json", metrics)
    joblib.dump({"model": model, "feature_columns": x.columns.tolist(),
                 "config": vars(args) | {"dataset_dir": str(args.dataset_dir),
                                         "output_dir": str(args.output_dir)},
                 "threshold": threshold},
                args.output_dir / "model.joblib")
    pd.DataFrame(sentinel).to_csv(args.output_dir / "leakage_sentinel.csv",
                                  index=False)
    if len(trades_oof):
        trades_oof.to_csv(args.output_dir / "trades_may_oof.csv", index=False)
    if len(trades_hold):
        trades_hold.to_csv(args.output_dir / "trades_july.csv", index=False)
    meta_out = meta_hold.copy()
    meta_out["y_prob"] = hold_prob
    meta_out.to_csv(args.output_dir / "predictions_july_ticks.csv",
                    index=False)
    meta_oof_out = meta_train.copy()
    meta_oof_out["y_prob"] = oof_prob
    meta_oof_out.to_csv(args.output_dir / "predictions_may_oof_ticks.csv",
                        index=False)
    print(f"\nSaved artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()
