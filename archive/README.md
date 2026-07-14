# Archive

Nothing in here is deleted — everything is moved out of the way so the active parts of
the repo (`configs/strategies/`, `runs/`) reflect current work. All strategies stay in
`configs/strategies/` untouched; nothing strategy-related lives in this archive.

## `legacy_market_data/`

Market-data folders/files confirmed (by diffing actual filenames/timestamps, not just
by "nothing references it") to be exact duplicates or derived/log artifacts of data that
still lives in its original place elsewhere in the repo:

- `simready2/`, `simready3/` — 100% filename overlap with `simulator_ready_markets/`.
- `simulator_ready_markets_liquidity_all_20260519_x2/` and `_x2_sequential/` — same 320
  underlying market timestamps as `simulator_ready_markets_liquidity_all_20260519/`,
  just double-processed with `__pass1`/`__pass2` naming.
- `simulator_ready_markets_start60_eff30_min1_set2/3/4` — every file already exists in
  `simulator_ready_markets_start60_eff30_min1_all/`.
- `datasets/` — derived ML feature/training CSVs (named `*_smoke`), not raw market data.
- Stale reject-log CSVs — audit logs of what got rejected during a filter run, not
  market data itself.

Nothing under here is a unique copy of live-recorded market data. If in doubt about a
market-data folder in the future, diff filenames/timestamps against the folders that
stay in place before assuming it's safe to touch — see git history around 2026-07-07 for
the verification method.

## `legacy_models/`

`reward_model_v1furst` (typo'd name) and `*_smoke` model dirs from `models/`.

## `legacy_run_trajectories/<run_name>/trajectories/`

Per-tick trajectory CSVs pulled out of large `runs/` folders to reclaim space (trajectory
files are the bulk of every run's size). Each run's `summary.csv` /
`group_summary.csv` / `market_summary.csv` / `strategy_report.html` stayed in place
under `runs/<run_name>/` — the trajectories here are supplementary detail, not the only
record of what happened. If a report's sample plots or trajectory links look broken,
the underlying CSV is here, not gone.

Two folders that looked like similar backtest sweeps were deliberately *not* trimmed:
- `runs/liquidity_inspection/` — contains `raw_snapshots.jsonl`, which is captured
  live orderbook data, not a regenerable strategy trajectory. Left fully untouched.
- `runs/thirdbatch_randomstrategies/` — has no `summary.csv` or report anywhere, so
  trimming its trajectories would have deleted the only record of those runs. Left
  fully untouched.

`runs/live_strategy_suite/` and `runs/final_candidates_risk_override_*` were kept
completely as-is (not trimmed) since they're the live/paper-trading track record and
explicitly-named final results.
