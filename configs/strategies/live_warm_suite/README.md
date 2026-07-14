# live_warm_suite

Curated set for warm live paper runs (assembled 2026-07-11). Copies — originals
stay in their home folders:

| config | what it is |
|---|---|
| combo_capalloc_nearstrike_lottery_only_17 | **best config tested** (added 2026-07-12): lottery leg gated on BTC within 0.05% of strike — 1079.5 median in-sample, positive OOS |
| combo_capalloc_lottery_only_17 | unfiltered lottery leg — live A/B partner for the nearstrike filter |
| combo_slow_plus_1t | warm momentum champion {1t,60s,120s,240s,900s}, 939.2 median in-sample |
| combo_sub_1t_2t_60s_120s | previous champion, reference |
| down_bias_chaser_lottery_sidepot_80_20_pct | classic down-bias reference family |

Run (paper mode, warm momentum by default, market CSVs recorded, no trajectory files):

```
python scripts/live/live_strategy_suite.py \
  --strategy-folder configs/strategies/live_warm_suite \
  --balance-config configs/live_start60_reserve180_topup_below5_drawdown60_window20_balance_bands.yaml \
  --execution-mode paper \
  --trajectory-log-mode none
```

Market tick CSVs land in `runs/live_strategy_suite/<run_id>/market_data/` in the
same format as the backtest market CSVs (feed them through
`find_simulator_ready_markets.py` to grow the backtest set). Per-strategy
results are still summarized in `summary.csv`; only per-tick trajectory files
are skipped. Expect the "BTC warmup: seeded N samples" banner from the second
market onwards (the first market of a fresh run has no prior history yet).
