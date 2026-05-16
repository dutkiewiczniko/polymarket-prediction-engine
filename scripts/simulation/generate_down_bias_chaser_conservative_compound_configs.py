from pathlib import Path


OUTPUT_DIR = Path("configs/strategies/down_bias_chaser_conservative_compound")
BATCH_CONFIG = Path("configs/simulation_down_bias_chaser_conservative_compound.yaml")

TAKE_PROFIT_VALUES = [0.95, 0.975]
CHEAP_PRICE_VALUES = [0.001, 0.005, 0.01]

PROFILES = [
    {
        "name": "careful_small",
        "momentum_pct": 4.5,
        "entry_cap": 0.72,
        "cooldown_ticks": 8,
        "max_orders": 20,
        "cheap_scale_tokens": 10,
        "btc_bias_scale_tokens": 8,
        "momentum_tokens": 4,
        "btc_gap_abs": 8.0,
        "min_balance_for_scale": 95.0,
    },
    {
        "name": "careful_medium",
        "momentum_pct": 4.0,
        "entry_cap": 0.76,
        "cooldown_ticks": 7,
        "max_orders": 24,
        "cheap_scale_tokens": 12,
        "btc_bias_scale_tokens": 10,
        "momentum_tokens": 5,
        "btc_gap_abs": 10.0,
        "min_balance_for_scale": 100.0,
    },
    {
        "name": "careful_patient",
        "momentum_pct": 5.0,
        "entry_cap": 0.68,
        "cooldown_ticks": 10,
        "max_orders": 16,
        "cheap_scale_tokens": 8,
        "btc_bias_scale_tokens": 6,
        "momentum_tokens": 3,
        "btc_gap_abs": 12.0,
        "min_balance_for_scale": 100.0,
    },
]


def clean_number(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def yaml_condition(metric: str, operator: str, value, indent: int = 8) -> str:
    pad = " " * indent
    return (
        f"{pad}- metric: {metric}\n"
        f"{pad}  operator: \"{operator}\"\n"
        f"{pad}  value: {value}\n"
    )


def rule_block(name: str, conditions: list[tuple[str, str, object]], action: str, amount_lines: list[str] | str | None = None) -> str:
    text = f"    - name: {name}\n      all:\n"
    for metric, operator, value in conditions:
        text += yaml_condition(metric, operator, value)
    text += f"      action: {action}\n"
    if amount_lines:
        if isinstance(amount_lines, str):
            amount_lines = [amount_lines]
        for line in amount_lines:
            text += f"      {line}\n"
    return text


BALANCE_SCALE_BANDS = [
    ("very_low_balance", None, 100.0, 40),
    ("low_balance", 100.0, 1000.0, 50),
    ("mid_balance", 1000.0, 2000.0, 100),
    ("high_balance", 2000.0, 3000.0, 200),
    ("very_high_balance", 3000.0, 4000.0, 300),
    ("ultra_balance", 4000.0, None, 400),
]


def balance_tiered_rules(
    base_name: str,
    conditions: list[tuple[str, str, object]],
    action: str,
    token_basis: int,
) -> str:
    rules: list[str] = []
    for label, min_balance, max_balance, cap in BALANCE_SCALE_BANDS:
        tier_conditions = list(conditions)
        if min_balance is not None:
            tier_conditions.append(("current_balance", ">=", clean_number(min_balance)))
        if max_balance is not None:
            tier_conditions.append(("current_balance", "<", clean_number(max_balance)))
        amount_lines = [
            f"balance_scaled_token_amount: {token_basis}",
            f"balance_scale_cap: {cap}",
        ]
        rules.append(rule_block(f"{base_name}_{label}", tier_conditions, action, amount_lines))
    return "\n\n".join(rules)


def take_profit_rules(take_profit: float) -> str:
    return (
        rule_block(
            "take_profit_up_position",
            [("has_up_position", "is", "true"), ("up_price", ">=", clean_number(take_profit))],
            "sell_up",
        )
        + "\n"
        + rule_block(
            "take_profit_down_position",
            [("has_down_position", "is", "true"), ("down_price", ">=", clean_number(take_profit))],
            "sell_down",
        )
    )


def cheap_rules(cheap_price: float, cheap_scale_tokens: int) -> str:
    return (
        rule_block(
            "cheap_up_probe_small",
            [("up_price", "<=", clean_number(cheap_price))],
            "buy_up",
            ["token_amount: 2"],
        )
        + "\n"
        + balance_tiered_rules(
            "cheap_down_scaled_entry",
            [("down_price", "<=", clean_number(cheap_price)), ("current_balance", ">=", clean_number(95.0))],
            "buy_down",
            cheap_scale_tokens,
        )
        + "\n"
        + rule_block(
            "cheap_down_probe_small",
            [("down_price", "<=", clean_number(cheap_price))],
            "buy_down",
            ["token_amount: 2"],
        )
    )


def btc_down_bias_rules(entry_cap: float, btc_gap_abs: float, min_balance_for_scale: float, scale_tokens: int) -> str:
    return (
        balance_tiered_rules(
            "btc_below_price_to_beat_scaled_down",
            [
                ("btc_below_price_to_beat", "is", "true"),
                ("btc_distance_to_price_to_beat", "<=", f"-{clean_number(btc_gap_abs)}"),
                ("down_price", "<=", clean_number(entry_cap)),
                ("current_balance", ">=", clean_number(min_balance_for_scale)),
            ],
            "buy_down",
            scale_tokens,
        )
        + "\n"
        + rule_block(
            "btc_below_price_to_beat_probe_down",
            [
                ("btc_below_price_to_beat", "is", "true"),
                ("btc_distance_to_price_to_beat", "<=", f"-{clean_number(max(4.0, btc_gap_abs * 0.5))}"),
                ("down_price", "<=", clean_number(entry_cap)),
            ],
            "buy_down",
            ["token_amount: 3"],
        )
    )


def chaser_rules(momentum_pct: float, entry_cap: float, momentum_tokens: int) -> str:
    return (
        rule_block(
            "up_price_momentum_probe_up",
            [("up_price_pct_change", ">=", clean_number(momentum_pct)), ("up_price", "<=", clean_number(entry_cap))],
            "buy_up",
            f"token_amount: {momentum_tokens}",
        )
        + "\n"
        + rule_block(
            "down_price_momentum_probe_down",
            [("down_price_pct_change", ">=", clean_number(momentum_pct)), ("down_price", "<=", clean_number(entry_cap))],
            "buy_down",
            f"token_amount: {momentum_tokens}",
        )
    )


def build_strategy_name(profile_name: str, take_profit: float, cheap_price: float) -> str:
    return (
        f"down_bias_chaser_{profile_name}"
        f"_tp{clean_number(take_profit).replace('.', '')}"
        f"_cheap{clean_number(cheap_price).replace('.', '')}"
    )


def build_strategy_yaml(profile: dict, take_profit: float, cheap_price: float) -> str:
    name = build_strategy_name(profile["name"], take_profit, cheap_price)
    sections = [
        take_profit_rules(take_profit),
        cheap_rules(cheap_price, profile["cheap_scale_tokens"]),
        btc_down_bias_rules(
            profile["entry_cap"],
            profile["btc_gap_abs"],
            profile["min_balance_for_scale"],
            profile["btc_bias_scale_tokens"],
        ),
        chaser_rules(profile["momentum_pct"], profile["entry_cap"], profile["momentum_tokens"]),
    ]
    return (
        f"name: {name}\n"
        "type: rule_based\n"
        "starting_balance: 100\n"
        "order_usd: 1\n\n"
        "params:\n"
        "  default_usd_amount: 1\n"
        f"  max_orders: {profile['max_orders']}\n"
        f"  cooldown_ticks: {profile['cooldown_ticks']}\n\n"
        "  rules:\n"
        + "\n".join(sections)
    )


def build_batch_yaml(strategy_paths: list[Path]) -> str:
    lines = [
        "batch_id: down_bias_chaser_conservative_compound_suite",
        "markets_folder: simulator_ready_markets",
        "market_pattern: btc-updown-5m-*.csv",
        "output_root: runs",
        "max_markets:",
        "",
        "starting_balance: 100",
        "order_usd: 1",
        "final_outcome:",
        "compound_balance: true",
        "",
        "strategies:",
    ]
    for path in strategy_paths:
        lines.extend([
            f"  - config: {path.as_posix()}",
            "    count: 1",
            f"    label: {path.stem}",
            "",
        ])
    return "\n".join(lines).rstrip() + "\n"


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for existing in OUTPUT_DIR.glob("*.yaml"):
        existing.unlink()

    strategy_paths: list[Path] = []
    for profile in PROFILES:
        for take_profit in TAKE_PROFIT_VALUES:
            for cheap_price in CHEAP_PRICE_VALUES:
                yaml_text = build_strategy_yaml(profile, take_profit, cheap_price)
                name = yaml_text.splitlines()[0].split(": ", 1)[1]
                path = OUTPUT_DIR / f"{name}.yaml"
                path.write_text(yaml_text, encoding="utf-8")
                strategy_paths.append(path)

    BATCH_CONFIG.write_text(build_batch_yaml(strategy_paths), encoding="utf-8")
    print(f"Wrote {len(strategy_paths)} strategy configs to {OUTPUT_DIR}")
    print(f"Wrote batch config to {BATCH_CONFIG}")


if __name__ == "__main__":
    main()
