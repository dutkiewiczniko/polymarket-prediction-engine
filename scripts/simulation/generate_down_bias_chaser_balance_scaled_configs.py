from pathlib import Path


OUTPUT_DIR = Path("configs/strategies/down_bias_chaser_balance_scaled")
BATCH_CONFIG = Path("configs/simulation_down_bias_chaser_balance_scaled.yaml")

TAKE_PROFIT_VALUES = [0.95, 0.975, 0.99]
CHEAP_PRICE_VALUES = [0.001, 0.005, 0.01, 0.015]
BALANCE_SCALED_TOKEN_BASES = [20, 30, 40]

BRANCHES = [
    {
        "name": "down_bias_chaser",
        "momentum_pct": 3.0,
        "entry_cap": 0.85,
        "cooldown_ticks": 3,
        "max_orders": 65,
    },
    {
        "name": "down_bias_chaser_careful",
        "momentum_pct": 4.0,
        "entry_cap": 0.76,
        "cooldown_ticks": 6,
        "max_orders": 40,
    },
    {
        "name": "down_bias_chaser_careful_tight",
        "momentum_pct": 5.0,
        "entry_cap": 0.70,
        "cooldown_ticks": 8,
        "max_orders": 28,
    },
    {
        "name": "down_bias_chaser_careful_patient",
        "momentum_pct": 4.5,
        "entry_cap": 0.72,
        "cooldown_ticks": 10,
        "max_orders": 24,
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


def rule_block(name: str, conditions: list[tuple[str, str, object]], action: str, amount_line: str = "") -> str:
    text = f"    - name: {name}\n      all:\n"
    for metric, operator, value in conditions:
        text += yaml_condition(metric, operator, value)
    text += f"      action: {action}\n"
    if amount_line:
        text += f"      {amount_line}\n"
    return text


def amount_line(base_tokens: int) -> str:
    return f"balance_scaled_token_amount: {base_tokens}"


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


def cheap_rules(cheap_price: float, base_tokens: int) -> str:
    return (
        rule_block(
            "cheap_up_optional_lottery",
            [("up_price", "<=", clean_number(cheap_price))],
            "buy_up",
            amount_line(base_tokens),
        )
        + "\n"
        + rule_block(
            "cheap_down_down_bias_lottery",
            [("down_price", "<=", clean_number(cheap_price))],
            "buy_down",
            amount_line(base_tokens),
        )
    )


def btc_down_bias_rules(entry_cap: float, base_tokens: int) -> str:
    return rule_block(
        "btc_below_price_to_beat_buy_down",
        [("btc_below_price_to_beat", "is", "true"), ("down_price", "<=", clean_number(entry_cap))],
        "buy_down",
        amount_line(base_tokens),
    )


def chaser_rules(momentum_pct: float, entry_cap: float, base_tokens: int) -> str:
    return (
        rule_block(
            "up_price_momentum_chase_buy_up",
            [("up_price_pct_change", ">=", clean_number(momentum_pct)), ("up_price", "<=", clean_number(entry_cap))],
            "buy_up",
            amount_line(base_tokens),
        )
        + "\n"
        + rule_block(
            "down_price_momentum_chase_buy_down",
            [("down_price_pct_change", ">=", clean_number(momentum_pct)), ("down_price", "<=", clean_number(entry_cap))],
            "buy_down",
            amount_line(base_tokens),
        )
    )


def build_strategy_name(branch_name: str, take_profit: float, cheap_price: float, base_tokens: int) -> str:
    return (
        f"{branch_name}"
        f"_tp{clean_number(take_profit).replace('.', '')}"
        f"_cheap{clean_number(cheap_price).replace('.', '')}"
        f"_balstok{base_tokens}"
    )


def build_strategy_yaml(branch: dict, take_profit: float, cheap_price: float, base_tokens: int) -> str:
    name = build_strategy_name(branch["name"], take_profit, cheap_price, base_tokens)
    sections = [
        take_profit_rules(take_profit),
        cheap_rules(cheap_price, base_tokens),
        btc_down_bias_rules(branch["entry_cap"], base_tokens),
        chaser_rules(branch["momentum_pct"], branch["entry_cap"], base_tokens),
    ]
    return (
        f"name: {name}\n"
        "type: rule_based\n"
        "starting_balance: 100\n"
        "order_usd: 1\n\n"
        "params:\n"
        "  default_usd_amount: 1\n"
        f"  max_orders: {branch['max_orders']}\n"
        f"  cooldown_ticks: {branch['cooldown_ticks']}\n\n"
        "  rules:\n"
        + "\n".join(sections)
    )


def build_batch_yaml(strategy_paths: list[Path]) -> str:
    lines = [
        "batch_id: down_bias_chaser_balance_scaled_suite",
        "markets_folder: market_data",
        "market_pattern: btc-updown-5m-*.csv",
        "output_root: runs",
        "max_markets:",
        "",
        "starting_balance: 100",
        "order_usd: 1",
        "final_outcome:",
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
    for branch in BRANCHES:
        for take_profit in TAKE_PROFIT_VALUES:
            for cheap_price in CHEAP_PRICE_VALUES:
                for base_tokens in BALANCE_SCALED_TOKEN_BASES:
                    yaml_text = build_strategy_yaml(branch, take_profit, cheap_price, base_tokens)
                    name = yaml_text.splitlines()[0].split(": ", 1)[1]
                    path = OUTPUT_DIR / f"{name}.yaml"
                    path.write_text(yaml_text, encoding="utf-8")
                    strategy_paths.append(path)

    BATCH_CONFIG.write_text(build_batch_yaml(strategy_paths), encoding="utf-8")
    print(f"Wrote {len(strategy_paths)} strategy configs to {OUTPUT_DIR}")
    print(f"Wrote batch config to {BATCH_CONFIG}")


if __name__ == "__main__":
    main()
