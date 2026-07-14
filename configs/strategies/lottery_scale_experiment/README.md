# lottery_scale_experiment — does the filtered lottery scale with capital?

**Question (2026-07-14):** the old "more lottery budget hurts" result (lot33 437
< lot17 1079) was measured on the UNFILTERED lottery, where ~80% of tickets were
junk beyond 0.05% of the strike. The nearstrike filter removes the junk — can
the filtered lottery absorb much larger capital shares? Directly tests the
user's observation that "the ones that catch the cheap lottery flips with a lot
of capital tend to succeed."

Generator: `scripts/simulation/generate_lottery_scale_configs.py`
Baseline to beat: `configs/strategies/momentum_metrics_experiment/combos/combo_capalloc_nearstrike_lottery_only_17.yaml`
(replicated here verbatim as `lotscale_b17_t25.yaml` — must reproduce its numbers).

All configs are lottery-only (take-profit 0.95 pair + nearstrike cheap-lottery
pair), no momentum, no warmup needed.

## Axes

1. **Budget** (`lotscale_b{17,33,50,100}_t25`): lottery pool
   `max_pool_spend_pct` at 0.167/0.33/0.50/1.0, ticket fixed at 25 tokens.
   Only binds if fire count × $0.625 tickets can actually reach the cap.
2. **Ticket-scaled** (`lotscale_b33_t49, b50_t75, b100_t150`): token_amount
   proportional to budget (25 per 16.7% share) so the pool drains in the same
   ~number of max-size fires as baseline. The real "a lot of capital per flip"
   variant.
3. **Filter grid** (`lotgrid_d{02,03,05}_p{010,025,050}`): distance gate
   0.02/0.03/0.05 % × price cap 0.01/0.025/0.05, at baseline budget/ticket.
   `d05_p025` cell = the baseline config.

## Bench

`simulator_ready_official_clean_plus_live` (434 mkts = 310 May clean + 124 July
live), 14 groups × 30 sequential, offset 0, true outcomes via
`--true-outcomes-csv simulator_ready_official_clean_plus_live_true_outcomes.csv`,
balance config `live_start60_reserve180_topup_below5_drawdown60_window20_balance_bands.yaml`.
Groups ~1-10 ≈ May cohort, ~12-14 ≈ July live cohort (built-in fresh-data view).

Run: `runs/compare/lottery_scale_14x30/` (log: `runs/compare/lottery_scale_14x30.log`)
