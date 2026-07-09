# Rule-Based Strategies

Rule-based strategies let you tune trading logic from YAML instead of writing a new
Python class for every idea.

Example config:

```yaml
name: rule_based_example
type: rule_based
starting_balance: 100
order_usd: 1

params:
  default_usd_amount: 1
  max_orders: 30
  cooldown_ticks: 5

  rules:
    - name: up_price_jumped_buy_up
      when:
        metric: up_price_pct_change
        operator: ">="
        value: 5
      action: buy_up
      usd_amount: 2
```

Rules are checked from top to bottom. The first matching rule wins for that tick.

## Actions

Supported actions are:

- `hold`
- `buy_up`
- `buy_down`
- `sell_up`
- `sell_down`

## Operators

Supported operators are:

- `>`
- `>=`
- `<`
- `<=`
- `==`
- `!=`
- `is`
- `not`

## Metrics

Supported metrics are:

- `up_price`
- `down_price`
- `up_minus_down_price`
- `down_minus_up_price`
- `up_price_distance_from_even`
- `down_price_distance_from_even`
- `up_price_pct_change`
- `down_price_pct_change`
- `btc_price`
- `btc_pct_change`
- `momentum_1s`, `momentum_2s`, `momentum_3s`, `momentum_15s`, `momentum_30s`, `momentum_60s`, `momentum_120s`
- `up_price_momentum_1s`, `up_price_momentum_2s`, `up_price_momentum_3s`, `up_price_momentum_15s`, `up_price_momentum_30s`, `up_price_momentum_60s`, `up_price_momentum_120s`
- `down_price_momentum_1s`, `down_price_momentum_2s`, `down_price_momentum_3s`, `down_price_momentum_15s`, `down_price_momentum_30s`, `down_price_momentum_60s`, `down_price_momentum_120s`
- `momentum_1t`, `momentum_2t`, `momentum_3t` (exactly N recorded *ticks* back, not elapsed seconds -- much faster/noisier than `momentum_1s` since tick spacing is sub-second)
- `up_price_momentum_1t`, `up_price_momentum_2t`, `up_price_momentum_3t`
- `down_price_momentum_1t`, `down_price_momentum_2t`, `down_price_momentum_3t`
- `btc_distance_to_price_to_beat`
- `btc_distance_to_price_to_beat_pct`
- `abs_btc_distance_to_price_to_beat`
- `abs_btc_distance_to_price_to_beat_pct`
- `btc_above_price_to_beat`
- `btc_below_price_to_beat`
- `seconds_left`
- `elapsed`
- `cash`
- `current_balance`
- `up_tokens`
- `down_tokens`
- `has_up_position`
- `has_down_position`
- `up_position_value`
- `down_position_value`
- `orders_placed`

`up_price_pct_change`, `down_price_pct_change`, and `btc_pct_change` compare the
current tick to whatever the *previous tick* happened to be -- since recorded ticks
are well under 1 second apart, this is effectively noise, not a real momentum signal.
The `momentum_*s` / `up_price_momentum_*s` / `down_price_momentum_*s` metrics instead
look back by elapsed wall-clock time (e.g. `momentum_15s` compares the current BTC
price to the price ~15 real seconds ago), matching the actual size of the window named.
They return `None` if no sample exists that far back yet (e.g. the first few seconds
of a market).

## Multiple Conditions

Use `all` when one rule needs several conditions.

```yaml
- name: btc_close_above_threshold_buy_up
  all:
    - metric: btc_distance_to_price_to_beat_pct
      operator: ">="
      value: 0
    - metric: btc_distance_to_price_to_beat_pct
      operator: "<="
      value: 0.05
  action: buy_up
  usd_amount: 3
```

Use `any` when one rule should fire if at least one condition matches.

```yaml
- name: buy_up_on_price_or_btc_momentum
  any:
    - metric: up_price_pct_change
      operator: ">="
      value: 4
    - metric: btc_pct_change
      operator: ">="
      value: 0.03
  action: buy_up
  usd_amount: 2
```

## Guardrails

`max_orders` limits total executed trade events for one market replay.

`max_market_spend_usd` caps total buy notional for one market. If a matching
buy would exceed the remaining cap, the order is reduced to the remaining
budget. Once the cap is exhausted, further buys hold with reason
`max market spend reached`.

Buy rules can bypass `max_orders` and `max_market_spend_usd` when they are
explicitly marked as lottery/risk overrides. Use `risk_override_max_price` to
allow the override only while the side price is at or below that threshold, or
`ignore_risk_limits: true` to bypass those limits unconditionally.

`cooldown_ticks` prevents repeated buying every tick after a rule fires.

### Pools (per-rule spend budgets)

`params.pools` defines named budgets that individual buy rules can be tagged
into via `pool: <name>`, independent of `max_market_spend_usd`. This is for
"never let rule A eat rule B's money" scenarios -- e.g. reserving a fixed
allocation for a cheap-lottery leg so a more aggressive directional rule can
never spend it, or giving each of several momentum-window rules its own slice
so a high-frequency rule can't starve a rarer, higher-conviction one.

```yaml
params:
  pools:
    main:
      max_pool_spend_usd: 70   # fixed dollar cap
    lottery:
      max_pool_spend_pct: 0.2  # 20% of market_start_balance -- scales with compounding
  rules:
    - name: chase_down
      pool: main
      ...
    - name: cheap_down_lottery
      pool: lottery
      ...
```

- `max_pool_spend_usd`: fixed dollar cap for that pool, for the whole market.
- `max_pool_spend_pct`: cap as a fraction of `market_start_balance` (the
  effective per-market balance, which already reflects compounding/bands) --
  use this instead of a hardcoded dollar amount so the pool scales correctly
  as the account balance grows or shrinks. If both are set, the stricter
  (smaller) of the two wins.
- A rule tagged with a `pool` is governed *only* by that pool's cap, not
  `max_market_spend_usd` (rules without a `pool` tag are unaffected and keep
  using the legacy global cap).
- `risk_override_max_price` / `ignore_risk_limits` bypass a rule's pool cap
  the same way they bypass `max_market_spend_usd`.
- Pool caps are enforced per matching rule under plain first-match-wins
  semantics (whichever rule matches first each tick). They do **not**
  currently attribute spend correctly per-rule when combined via
  `combine_matching_buys` in the same tick -- for "each rule has its own
  pot," give each window/signal its own rule (and pool) rather than relying
  on combine_matching_buys to sum them.

`default_usd_amount` is used when a rule does not define `usd_amount`.

`sell_opposite_first` controls whether a buy closes the existing opposite-side
position before opening the new position. It defaults to `true` to preserve
existing behavior. Set it at `params` level for the whole rule strategy, or on
an individual buy rule:

```yaml
params:
  sell_opposite_first: false
  rules:
    - name: cheap_down_lottery_keep_up_ticket
      when:
        metric: down_price
        operator: "<="
        value: 0.025
      action: buy_down
      token_amount: 25
      sell_opposite_first: false
```

Buy rules can also define size in strategy terms:

- `usd_amount`: fixed USD notional.
- `balance_pct`: USD notional as a fraction of the market starting balance.
- `token_amount`: desired number of outcome tokens, converted to USD using the current side price.
- `balance_scaled_token_amount`: token amount scaled by `market_start_balance / 100`.

If more than one sizing field is present, later fields in that list override earlier
ones. For example, `token_amount` overrides `balance_pct`.

## Voting Ensemble Strategy

`type: voting_ensemble` runs several member strategy configs in parallel and only
trades when at least `min_votes` of them agree on the same side that tick.

```yaml
type: voting_ensemble
params:
  min_votes: 2
  default_scale: 0.75
  priority_scale: 0.65
  size_mode: max
  priority_members:
    - family_up_bias_chaser
  members:
    - label: family_up_bias_chaser
      config: configs/strategies/.../up_bias_chaser.yaml
    - label: family_neutral_chaser
      config: configs/strategies/.../neutral_chaser.yaml
```

When enough members agree, the trade is sized off one of the agreeing members'
requested `usd_amount`, then scaled by `priority_scale` (if any agreeing member is
in `priority_members`) or `default_scale` otherwise. `size_mode` picks which
agreeing order size to use:

- `min` (default): the smallest agreeing order size. Conservative.
- `max`: the largest agreeing order size. More aggressive; use
  `scripts/simulation/compare_strategies.py` (see `SCRIPTS_SUMMARY.md`) to check
  the tradeoff against `min` before switching a live config.
