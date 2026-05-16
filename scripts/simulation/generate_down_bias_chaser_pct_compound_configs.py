from pathlib import Path


OUTPUT_DIR = Path("configs/strategies/down_bias_chaser_pct_compound")
BATCH_CONFIG = Path("configs/simulation_down_bias_chaser_pct_compound.yaml")

TAKE_PROFIT_VALUES = [0.95, 0.975, 0.99]
CHEAP_PRICE_VALUES = [0.01, 0.015]
ORDER_CASH_PCTS = [0.10, 0.20]

ENTRY_CAP = 0.85
MOMENTUM_PCT = 3.0
COOLDOWN_TICKS = 3
MAX_ORDERS = 40


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


def cash_pct_rule(base_name: str, conditions: list[tuple[str, str, object]], action: str, cash_pct: float) -> str:
    return rule_block(base_name, conditions, action, [f"cash_usd_pct: {clean_number(cash_pct)}"])


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


def cheap_rules(cheap_price: float, cash_pct: float) -> str:
    return (
        cash_pct_rule(
            "cheap_up_optional_lottery",
            [("up_price", "<=", clean_number(cheap_price))],
            "buy_up",
            cash_pct,
        )
        + "\n"
        + cash_pct_rule(
            "cheap_down_down_bias_lottery",
            [("down_price", "<=", clean_number(cheap_price))],
            "buy_down",
            cash_pct,
        )
    )


def btc_down_bias_rules(cash_pct: float) -> str:
    return cash_pct_rule(
        "btc_below_price_to_beat_buy_down",
        [
            ("btc_below_price_to_beat", "is", "true"),
            ("btc_distance_to_price_to_beat", "<=", "-10"),
            ("down_price", "<=", clean_number(ENTRY_CAP)),
        ],
        "buy_down",
        cash_pct,
    )


def chaser_rules(cash_pct: float) -> str:
    return (
        cash_pct_rule(
            "up_price_momentum_chase_buy_up",
            [("up_price_pct_change", ">=", clean_number(MOMENTUM_PCT)), ("up_price", "<=", clean_number(ENTRY_CAP))],
            "buy_up",
            cash_pct,
        )
        + "\n"
        + cash_pct_rule(
            "down_price_momentum_chase_buy_down",
            [("down_price_pct_change", ">=", clean_number(MOMENTUM_PCT)), ("down_price", "<=", clean_number(ENTRY_CAP))],
            "buy_down",
            cash_pct,
        )
    )


def build_strategy_name(take_profit: float, cheap_price: float, cash_pct: float) -> str:
    return (
        f"down_bias_chaser"
        f"_tp{clean_number(take_profit).replace('.', '')}"
        f"_cheap{clean_number(cheap_price).replace('.', '')}"
        f"_cashpct{int(round(cash_pct * 100)):02d}"
    )


def build_strategy_yaml(take_profit: float, cheap_price: float, cash_pct: float) -> str:
    name = build_strategy_name(take_profit, cheap_price, cash_pct)
    sections = [
        take_profit_rules(take_profit),
        cheap_rules(cheap_price, cash_pct),
        btc_down_bias_rules(cash_pct),
        chaser_rules(cash_pct),
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
        "batch_id: down_bias_chaser_pct_compound_suite",
        "markets_folder: simulator_ready_markets",
        "market_pattern: btc-updown-5m-*.csv",
        "output_root: runs",
        "max_markets:",
        "",
        "starting_balance: 2000",
        "order_usd: 1",
        "final_outcome:",
        "compound_balance: true",
        "effective_market_balance_pct: 0.05",
        "effective_market_balance_min: 50",
        "effective_market_balance_max: 300",
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
    for take_profit in TAKE_PROFIT_VALUES:
        for cheap_price in CHEAP_PRICE_VALUES:
            for cash_pct in ORDER_CASH_PCTS:
                yaml_text = build_strategy_yaml(take_profit, cheap_price, cash_pct)
                name = yaml_text.splitlines()[0].split(": ", 1)[1]
                path = OUTPUT_DIR / f"{name}.yaml"
                path.write_text(yaml_text, encoding="utf-8")
                strategy_paths.append(path)

    BATCH_CONFIG.write_text(build_batch_yaml(strategy_paths), encoding="utf-8")
    print(f"Wrote {len(strategy_paths)} strategy configs to {OUTPUT_DIR}")
    print(f"Wrote batch config to {BATCH_CONFIG}")


if __name__ == "__main__":
    main()
