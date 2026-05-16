from pathlib import Path


OUTPUT_DIR = Path("configs/strategies/down_bias_chaser_capped_token_compound")
BATCH_CONFIG = Path("configs/simulation_down_bias_chaser_capped_token_compound.yaml")

TAKE_PROFIT_VALUES = [0.95, 0.975, 0.99]
CHEAP_PRICE_VALUES = [0.01, 0.015]
TOKEN_AMOUNTS = [10, 20]

ENTRY_CAP = 0.85
MOMENTUM_PCT = 3.0
COOLDOWN_TICKS = 3
MAX_ORDERS = 65
DOWN_TOKEN_MULTIPLIER = 1.5

# Responsive master-balance bands:
# below 1000   -> simulate market with $50, no token increase
# 1000-2999.99 -> simulate market with $100, base tokens
# 3000-3999.99 -> simulate market with $200, base tokens + 10
# 4000-4999.99 -> simulate market with $300, base tokens + 20
# and so on.
BALANCE_RESPONSE_BANDS = [
    ("under_1000", None, 1000.0, 50, 0),
    ("1000_to_3000", 1000.0, 3000.0, 100, 0),
    ("3000_to_4000", 3000.0, 4000.0, 200, 10),
    ("4000_to_5000", 4000.0, 5000.0, 300, 20),
    ("5000_to_6000", 5000.0, 6000.0, 400, 30),
    ("6000_to_7000", 6000.0, 7000.0, 500, 40),
    ("7000_plus", 7000.0, None, 600, 50),
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


def tiered_token_rules(base_name: str, conditions: list[tuple[str, str, object]], action: str, token_amount: int) -> str:
    rules: list[str] = []
    for label, min_balance, max_balance, _simulated_balance, token_bonus in BALANCE_RESPONSE_BANDS:
        adjusted_tokens = token_amount + token_bonus
        tier_conditions = list(conditions)
        if min_balance is not None:
            tier_conditions.append(("market_start_balance", ">=", clean_number(min_balance)))
        if max_balance is not None:
            tier_conditions.append(("market_start_balance", "<", clean_number(max_balance)))
        rules.append(
            rule_block(
                f"{base_name}_{label}",
                tier_conditions,
                action,
                [f"token_amount: {adjusted_tokens}"],
            )
        )
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


def cheap_rules(cheap_price: float, token_amount: int) -> str:
    down_tokens = max(1, int(round(token_amount * DOWN_TOKEN_MULTIPLIER)))
    return (
        tiered_token_rules(
            "cheap_up_optional_lottery",
            [("up_price", "<=", clean_number(cheap_price))],
            "buy_up",
            token_amount,
        )
        + "\n"
        + tiered_token_rules(
            "cheap_down_down_bias_lottery",
            [("down_price", "<=", clean_number(cheap_price))],
            "buy_down",
            down_tokens,
        )
    )


def btc_down_bias_rules(token_amount: int) -> str:
    down_tokens = max(1, int(round(token_amount * DOWN_TOKEN_MULTIPLIER)))
    return tiered_token_rules(
        "btc_below_price_to_beat_buy_down",
        [("btc_below_price_to_beat", "is", "true"), ("down_price", "<=", clean_number(ENTRY_CAP))],
        "buy_down",
        down_tokens,
    )


def chaser_rules(token_amount: int) -> str:
    down_tokens = max(1, int(round(token_amount * DOWN_TOKEN_MULTIPLIER)))
    return (
        tiered_token_rules(
            "up_price_momentum_chase_buy_up",
            [("up_price_pct_change", ">=", clean_number(MOMENTUM_PCT)), ("up_price", "<=", clean_number(ENTRY_CAP))],
            "buy_up",
            token_amount,
        )
        + "\n"
        + tiered_token_rules(
            "down_price_momentum_chase_buy_down",
            [("down_price_pct_change", ">=", clean_number(MOMENTUM_PCT)), ("down_price", "<=", clean_number(ENTRY_CAP))],
            "buy_down",
            down_tokens,
        )
    )


def build_strategy_name(take_profit: float, cheap_price: float, token_amount: int) -> str:
    return (
        f"down_bias_chaser"
        f"_tp{clean_number(take_profit).replace('.', '')}"
        f"_cheap{clean_number(cheap_price).replace('.', '')}"
        f"_tok{token_amount}"
    )


def build_strategy_yaml(take_profit: float, cheap_price: float, token_amount: int) -> str:
    name = build_strategy_name(take_profit, cheap_price, token_amount)
    sections = [
        take_profit_rules(take_profit),
        cheap_rules(cheap_price, token_amount),
        btc_down_bias_rules(token_amount),
        chaser_rules(token_amount),
    ]
    return (
        f"name: {name}\n"
        "type: rule_based\n"
        "starting_balance: 2000\n"
        "order_usd: 1\n\n"
        "params:\n"
        "  default_usd_amount: 1\n"
        f"  max_orders: {MAX_ORDERS}\n"
        f"  cooldown_ticks: {COOLDOWN_TICKS}\n\n"
        "  rules:\n"
        + "\n".join(sections)
    )


def build_batch_yaml(strategy_paths: list[Path]) -> str:
    lines = [
        "batch_id: down_bias_chaser_capped_token_compound_suite",
        "markets_folder: simulator_ready_markets",
        "market_pattern: btc-updown-5m-*.csv",
        "output_root: runs",
        "max_markets:",
        "",
        "starting_balance: 2000",
        "order_usd: 1",
        "final_outcome:",
        "compound_balance: true",
        "effective_market_balance_bands:",
    ]
    for _label, min_balance, max_balance, simulated_balance, _token_bonus in BALANCE_RESPONSE_BANDS:
        lines.append("  -")
        if min_balance is not None:
            lines.append(f"    min_total_balance: {clean_number(min_balance)}")
        if max_balance is not None:
            lines.append(f"    max_total_balance: {clean_number(max_balance)}")
        lines.append(f"    simulated_balance: {simulated_balance}")
    lines.extend([
        "",
        "strategies:",
    ])
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
    for take_profit in TAKE_PROFIT_VALUES:
        for cheap_price in CHEAP_PRICE_VALUES:
            for token_amount in TOKEN_AMOUNTS:
                yaml_text = build_strategy_yaml(take_profit, cheap_price, token_amount)
                name = yaml_text.splitlines()[0].split(": ", 1)[1]
                path = OUTPUT_DIR / f"{name}.yaml"
                path.write_text(yaml_text, encoding="utf-8")
                strategy_paths.append(path)

    BATCH_CONFIG.write_text(build_batch_yaml(strategy_paths), encoding="utf-8")
    print(f"Wrote {len(strategy_paths)} strategy configs to {OUTPUT_DIR}")
    print(f"Wrote batch config to {BATCH_CONFIG}")


if __name__ == "__main__":
    main()
