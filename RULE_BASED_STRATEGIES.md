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

`cooldown_ticks` prevents repeated buying every tick after a rule fires.

`default_usd_amount` is used when a rule does not define `usd_amount`.
