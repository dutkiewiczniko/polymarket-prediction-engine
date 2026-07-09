"""Generate momentum window-combination strategy configs.

Each generated strategy follows the pattern validated in the momentum_combo
experiments: one buy_up/buy_down rule pair per momentum window, each pair with
its own pct-based pool so fast high-frequency windows can't starve slow rare
ones, rules ordered slowest-to-fastest (first-match-wins gives the rarer,
higher-conviction signal priority), plus a shared cheap-lottery pool and
take-profit sells.

Window vocabulary: Nt = N ticks back, Ns = N seconds back.
"""

from pathlib import Path

OUTPUT_DIR = Path("configs/strategies/momentum_metrics_experiment/combos")

# Threshold per window, calibrated from real market data (see momentum
# experiments: tick/1-3s windows are ~0 most ticks -> near-zero epsilon;
# longer windows use ~p75-p90 of their observed absolute magnitude).
WINDOW_THRESHOLDS = {
    "1t": 0.0001,
    "2t": 0.0001,
    "3t": 0.0001,
    "4t": 0.0001,
    "1s": 0.0001,
    "2s": 0.0005,
    "3s": 0.001,
    "15s": 0.01,
    "30s": 0.015,
    "60s": 0.03,
    "120s": 0.03,
    # New grid windows: BTC magnitude at these scales grows ~sqrt(window)
    # (confirmed against calibration: p75 |momentum| ratio 15s->30s was 1.43 ~ sqrt(2)),
    # so threshold(W) ~ 0.01 * sqrt(W/15), rounded.
    "5s": 0.006,
    "10s": 0.008,
    "20s": 0.012,
    "40s": 0.016,
    "80s": 0.023,
    "160s": 0.033,
    "230s": 0.039,
    "240s": 0.04,
    "250s": 0.041,
    "275s": 0.043,
    # Beyond one market's lifetime: only usable with --warmup-prior-market.
    "300s": 0.045,
    "420s": 0.055,
    "600s": 0.065,
    "900s": 0.08,
    # 1200s = 4 markets back, 1500s = 5 markets back (each market = 300s).
    "1200s": 0.09,
    "1500s": 0.10,
}

# Sort key: slowest first. Seconds windows are slower than tick windows of
# similar nominal number because ticks are ~0.2s apart.
def window_speed(window: str) -> float:
    value = float(window[:-1])
    if window.endswith("t"):
        return value * 0.2
    return value


COMBOS = {
    "all10": ["1t", "2t", "3t", "1s", "2s", "3s", "15s", "30s", "60s", "120s"],
    "ticks_only": ["1t", "2t", "3t"],
    "1t_2t": ["1t", "2t"],
    "1t_3t": ["1t", "3t"],
    # prev_winner_6w was [1t,2t,1s,15s,60s,120s]; this adds the load-bearing 3t.
    "prev6_plus_3t": ["1t", "2t", "3t", "1s", "15s", "60s", "120s"],
    "prev6_plus_3t_no120": ["1t", "2t", "3t", "1s", "15s", "60s"],
    "ticks4": ["1t", "2t", "3t", "4t"],
    # My suggestion: only the two tiers that tested strong (ticks + slow),
    # dropping the weak mid (15s/30s) and the redundant fast-second (1s).
    "ticks_slow": ["1t", "2t", "3t", "60s", "120s"],
    # Leave-one-out ablation of prev_winner_6w [1t,2t,1s,15s,60s,120s]:
    # each config removes exactly one window. Comparing each against the full
    # winner measures that window's net importance (signal lost vs pool share
    # freed for the survivors).
    "loo_no_1t": ["2t", "1s", "15s", "60s", "120s"],
    "loo_no_2t": ["1t", "1s", "15s", "60s", "120s"],
    "loo_no_1s": ["1t", "2t", "15s", "60s", "120s"],
    "loo_no_15s": ["1t", "2t", "1s", "60s", "120s"],
    "loo_no_60s": ["1t", "2t", "1s", "15s", "120s"],
    "loo_no_120s": ["1t", "2t", "1s", "15s", "60s"],
    # Champion {1t,2t,60s,120s} with the 2t->3t swap: 3t beat 2t decisively in
    # pairs (1t_3t 777 vs 1t_2t 287) but was scoped out of the subset search,
    # which only covered the old winner's vocabulary.
    "champ_2t_to_3t": ["1t", "3t", "60s", "120s"],
    "seconds_fast_only": ["1s", "2s", "3s"],
    "mid_only": ["15s", "30s"],
    "slow_only": ["60s", "120s"],
    "fast_mid": ["1t", "1s", "15s"],
    "fast_slow": ["1t", "1s", "60s", "120s"],
    "mid_slow": ["15s", "30s", "60s", "120s"],
    "one_per_tier": ["1t", "1s", "15s", "60s"],
    "prev_winner_6w": ["1t", "2t", "1s", "15s", "60s", "120s"],
    "prev_3w": ["1t", "1s", "30s"],
    "pair_1t_120s": ["1t", "120s"],
    "pair_1s_15s": ["1s", "15s"],
    "pair_15s_60s": ["15s", "60s"],
    "no_ticks": ["1s", "2s", "3s", "15s", "30s", "60s", "120s"],
    "no_seconds_fast": ["1t", "2t", "3t", "15s", "30s", "60s", "120s"],
    # Extending the champion {1t,2t,60s,120s} with an ultra-long window to
    # test whether it improves the weak-regime (group 3) floor.
    "champ_plus_900s": ["1t", "2t", "60s", "120s", "900s"],
    "champ_240_900": ["1t", "2t", "60s", "240s", "900s"],
}


def momentum_rule_pair(window: str, threshold: float) -> str:
    pool = f"momentum_{window}"
    return f"""    - name: momentum_{window}_buy_up
      pool: {pool}
      all:
        - metric: momentum_{window}
          operator: ">="
          value: {threshold}
        - metric: up_price
          operator: "<="
          value: 0.85
      action: buy_up
      token_amount: 10

    - name: momentum_{window}_buy_down
      pool: {pool}
      all:
        - metric: momentum_{window}
          operator: "<="
          value: -{threshold}
        - metric: down_price
          operator: "<="
          value: 0.85
      action: buy_down
      token_amount: 10
"""


def build_config(combo_name: str, windows: list[str]) -> str:
    ordered = sorted(windows, key=window_speed, reverse=True)
    n_pools = len(ordered) + 1  # +1 for the lottery pool
    share = round(1.0 / n_pools, 6)

    pool_lines = ["    lottery:", f"      max_pool_spend_pct: {share}"]
    for window in ordered:
        pool_lines.append(f"    momentum_{window}:")
        pool_lines.append(f"      max_pool_spend_pct: {share}")
    pools_block = "\n".join(pool_lines)

    rules_blocks = "\n".join(
        momentum_rule_pair(window, WINDOW_THRESHOLDS[window]) for window in ordered
    )

    return f"""name: momentum_combo_{combo_name}
type: rule_based
starting_balance: 100
order_usd: 1

# Auto-generated by scripts/simulation/generate_momentum_combo_configs.py
# Windows (slowest-to-fastest): {', '.join(ordered)}
# Each window has its own pct pool ({share:.1%} of effective balance each,
# lottery included), so no rule can starve another's budget.

params:
  default_usd_amount: 1
  max_orders: 200
  cooldown_ticks: 3

  pools:
{pools_block}

  rules:
    - name: take_profit_up_position
      all:
        - metric: has_up_position
          operator: "is"
          value: true
        - metric: up_price
          operator: ">="
          value: 0.95
      action: sell_up

    - name: take_profit_down_position
      all:
        - metric: has_down_position
          operator: "is"
          value: true
        - metric: down_price
          operator: ">="
          value: 0.95
      action: sell_down

    - name: cheap_up_lottery
      pool: lottery
      all:
        - metric: up_price
          operator: "<="
          value: 0.025
      action: buy_up
      token_amount: 25

    - name: cheap_down_lottery
      pool: lottery
      all:
        - metric: down_price
          operator: "<="
          value: 0.025
      action: buy_down
      token_amount: 25

{rules_blocks}"""


# The reigning best combo. All 2^6 subsets of it get generated (sub_* configs)
# so the full combination space can be searched exhaustively and each window's
# importance measured across every context (Shapley-style), not just via greedy
# leave-one-out drops.
WINNER_SET = ["1t", "2t", "1s", "15s", "60s", "120s"]


# Stage A of the period-axis search: single-window configs across a log-spaced
# grid, to map where on the time axis the momentum signal actually lives
# (instead of hand-picking round-number windows).
RESPONSE_CURVE_WINDOWS = [
    "1t", "2t", "3t", "4t",
    "1s", "2s", "3s", "5s", "10s", "15s", "20s", "30s", "40s", "60s", "80s", "120s", "160s",
    "230s", "240s", "250s", "275s",
    "300s", "420s", "600s", "900s", "1200s", "1500s",
]


def response_curve_singles() -> dict[str, list[str]]:
    return {f"single_{w}": [w] for w in RESPONSE_CURVE_WINDOWS}


def all_winner_subsets() -> dict[str, list[str]]:
    from itertools import combinations

    subsets = {}
    for size in range(0, len(WINNER_SET) + 1):
        for combo in combinations(WINNER_SET, size):
            name = "sub_" + "_".join(combo) if combo else "sub_none"
            subsets[name] = list(combo)
    return subsets


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for combo_name, windows in {**COMBOS, **all_winner_subsets(), **response_curve_singles()}.items():
        unknown = [w for w in windows if w not in WINDOW_THRESHOLDS]
        if unknown:
            raise ValueError(f"{combo_name}: unknown windows {unknown}")
        path = OUTPUT_DIR / f"combo_{combo_name}.yaml"
        path.write_text(build_config(combo_name, windows), encoding="utf-8")
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
