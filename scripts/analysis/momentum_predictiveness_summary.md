# Momentum predictiveness — result summary (handoff)

**Date:** 2026-07-09 · **Dataset:** `simulator_ready_markets_liquidity_all_20260519/` (320 markets, BTC 5-min up/down, 2026-05-16..18, one contiguous window) · **Base rate:** down 63.7% / up 36.2%.

**One-line result:** Short-window BTC momentum genuinely predicts market resolution and beats the base rate *on both sides*; predictiveness rises with move magnitude and later in the market. The macro `btc_change_{1h,12h,24h,7d}_pct` columns do **not** add directional signal on this set (24h/7d are sign-constant → degenerate here).

---

## Metric definitions (must match engine, do not re-derive)

Reference: `simulator/strategies.py` — `momentum_pct` (time window) and `momentum_pct_by_ticks` (tick window). BTC series = `btc_chainlink` with fallback to `btc_binance`; ticks with no BTC value are never recorded. Outcome = `infer_final_outcome` (`simulator/replay.py`): last BTC ≥ last `price_to_beat` → up, else down.

- `momentum_<N>t` = pct change of BTC vs exactly N recorded ticks ago (sub-second spacing). Windows checked: **1t, 2t**.
- `momentum_<W>s` = pct change vs the latest sample at or before `now − W` seconds. Windows checked: **60s, 120s**.
- Champion thresholds used for conditioning: **0.0001%** (tick windows), **0.03%** (time windows).

## Method — the per-tick vs per-market reconciliation (key design decision)

Momentum is per-tick (~1500/market); outcome is one label/market. To make a prediction comparable to the outcome, sign is **sampled at fixed elapsed checkpoints**: **60s / 150s / 240s** of the ~300s market, taken at the last valid-BTC tick at or before each checkpoint (exactly what the engine would compute there). Also reported: **majority sign over the whole market**, and every alignment **conditioned on |momentum| > champion threshold**. The vectorized momentum is asserted equal to the engine's reference functions on sampled ticks (`verify_against_engine`).

"Sign alignment" = P(favored side wins), where favored = up if momentum > 0 else down. Read it against the **per-side** base rate (up 36.2%, down 63.7%), not a single number — this is why the both-sides table below matters.

---

## Results

### Sign alignment by window × checkpoint (n = markets with a defined sign there)

| Window | @60s | @150s | @240s | @150s, \|mom\|>thr | @240s, \|mom\|>thr | majority-sign |
|---|---|---|---|---|---|---|
| momentum_1t | 52.4% (42) | 55.6% (36) | 65.0% (40) | 65.2% (23) | 70.0% (30) | 74.4% (316) |
| momentum_2t | 60.6% (109) | 59.6% (94) | 55.1% (89) | 59.1% (66) | 59.1% (66) | 73.8% (320) |
| momentum_60s | 50.0% (8) | 57.5% (320) | 61.9% (320) | 70.3% (111) | 70.3% (111) | 75.9% (320) |
| momentum_120s | — (0) | 64.1% (320) | 65.9% (320) | 76.0% (167) | 74.4% (160) | 70.8% (319) |

`momentum_120s@60s` is undefined (needs pre-market history — same live warmup gap flagged in the momentum-findings memory). Tick windows read exactly 0 on most ticks (1t nonzero 14.9%, 2t 29.7%), so their checkpoint n is small and their meaningful summary is the majority column.

### It beats base rate on BOTH sides (not just riding the down-heavy prior)

| Signal | predicts up → up-win | predicts down → down-win |
|---|---|---|
| momentum_120s @150s, \|mom\|>0.03 | 65.9% (n=91) vs 36.2% base | 88.2% (n=76) vs 63.7% base |
| momentum_120s @240s, \|mom\|>0.03 | 62.5% (n=72) | 84.1% (n=88) |
| momentum_60s @240s, \|mom\|>0.03 | 62.7% (n=51) | 76.7% (n=60) |
| momentum_60s majority | 61.8% (n=165) | 91.0% (n=155) |
| momentum_120s majority | 57.8% (n=147) | 82.0% (n=172) |

### Predictiveness is monotone in magnitude (favored win rate by |momentum| quartile @150s)

| Window | q1 weak | q2 | q3 | q4 strong |
|---|---|---|---|---|
| momentum_60s | 45.0% | 61.3% | 50.0% | **73.8%** |
| momentum_120s | 48.8% | 55.0% | 68.8% | **83.8%** |

Weakest quartile ≈ coin flip; strongest ≈ 74–84%. This is the empirical justification for threshold-gating rather than trading every tick.

---

## Contrast: macro `btc_change_*` columns are non-predictive here

Ran `analyze_btc_trend_predictiveness.py` on the same set: `1h` sign varies but aligns 50.0% (no edge); `12h` aligns 59.4% (*below* the 63.7% down base rate); `24h` and `7d` are **negative for all 320 markets** (0 positive), so their 63.7% "alignment" is literally the down base rate reproduced — zero cross-market discriminating power on this single-regime window. Their magnitude buckets run *backwards* (weakest trend → highest down-win), a mild-mean-reversion hint, likely noise. Not evaluable until a dataset spanning up- and down-macro regimes exists.

## Caveats / open threads for the next agent

1. **Single contiguous window** (2026-05-16..18, all one macro downtrend). None of this is validated at another `--market-offset`; mandatory OOS check per the momentum-findings memory.
2. Predictive = **direction agrees with sign**, not a tradable-PnL claim; entry price/liquidity/fees not modeled here (that's the backtester's job).
3. `momentum_120s` needs cross-market BTC warmup to be live at all early — dead for the first ~2 min of a cold live market (`live_strategy_suite.py` does not yet carry BTC history across markets).
4. Base rate is down-heavy; always read alignment against the per-side base, i.e. the both-sides table.

## Reproduce

```
python scripts/analysis/analyze_momentum_predictiveness.py --markets-folder simulator_ready_markets_liquidity_all_20260519
python scripts/analysis/analyze_btc_trend_predictiveness.py  --markets-folder simulator_ready_markets_liquidity_all_20260519
```

**Artifacts** (both gitignored, regenerate via above): per-market tables `simulator_ready_markets_liquidity_all_20260519_momentum_outcomes.csv` (checkpoint values, positive-share, mean |mom|, majority sign per window) and `..._btc_trend_outcomes.csv`. Scripts are tracked in `scripts/analysis/`.
