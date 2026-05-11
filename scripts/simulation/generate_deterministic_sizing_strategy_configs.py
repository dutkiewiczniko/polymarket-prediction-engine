from pathlib import Path


OUTPUT_DIR = Path("configs/strategies/deterministic_diverse_sizing")
BATCH_CONFIG = Path("configs/simulation_deterministic_diverse_sizing.yaml")

TAKE_PROFIT_VALUES = [0.75, 0.80, 0.90, 0.95]
CHEAP_PRICE_VALUES = [0.015, 0.025, 0.05]
TOKEN_AMOUNT_VALUES = [30]
BALANCE_PCT_VALUES = [0.10, 0.30]

BRANCHES = [
    {
        "name": "down_bias_chaser",
        "momentum_pct": 3.0,
        "spike_pct": 6.0,
        "entry_cap": 0.85,
        "cooldown_ticks": 3,
        "max_orders": 65,
        "down_size_multiplier": 1.5,
        "rules": ["take_profit", "cheap", "btc_down_bias", "chaser"],
    },
    {
        "name": "down_bias_chaser_careful",
        "momentum_pct": 4.5,
        "spike_pct": 6.5,
        "entry_cap": 0.75,
        "cooldown_ticks": 6,
        "max_orders": 40,
        "down_size_multiplier": 1.35,
        "rules": ["take_profit", "cheap", "btc_down_bias", "chaser"],
    },
    {
        "name": "down_bias_fader",
        "momentum_pct": 4.0,
        "spike_pct": 5.0,
        "entry_cap": 0.90,
        "cooldown_ticks": 5,
        "max_orders": 55,
        "down_size_multiplier": 1.5,
        "rules": ["take_profit", "cheap", "btc_down_bias", "fader"],
    },
    {
        "name": "down_bias_fader_careful",
        "momentum_pct": 4.5,
        "spike_pct": 6.5,
        "entry_cap": 0.80,
        "cooldown_ticks": 8,
        "max_orders": 35,
        "down_size_multiplier": 1.35,
        "rules": ["take_profit", "cheap", "btc_down_bias", "fader"],
    },
    {
        "name": "hybrid_price_signal",
        "momentum_pct": 2.5,
        "spike_pct": 5.0,
        "entry_cap": 0.90,
        "cooldown_ticks": 4,
        "max_orders": 70,
        "down_size_multiplier": 1.25,
        "rules": ["take_profit", "cheap", "btc_directional", "chaser", "fader"],
    },
    {
        "name": "hybrid_price_signal_careful",
        "momentum_pct": 3.75,
        "spike_pct": 6.0,
        "entry_cap": 0.78,
        "cooldown_ticks": 7,
        "max_orders": 42,
        "down_size_multiplier": 1.2,
        "rules": ["take_profit", "cheap", "btc_directional", "chaser", "fader"],
    },
]


def clean_number(value: float) -> str:
    return f"{value:.3f}".rstrip("0").rstrip(".")


def clean_pct_code(value: float) -> str:
    return str(int(round(value * 100)))


def yaml_condition(metric: str, operator: str, value, indent: int = 8) -> str:
    pad = " " * indent
    return (
        f"{pad}- metric: {metric}\n"
        f"{pad}  operator: \"{operator}\"\n"
        f"{pad}  value: {value}\n"
    )


def rule_block(name: str, conditions: list[tuple[str, str, object]], action: str, amount_line: str) -> str:
    text = f"    - name: {name}\n      all:\n"
    for metric, operator, value in conditions:
        text += yaml_condition(metric, operator, value)
    text += f"      action: {action}\n"
    if amount_line:
        text += f"      {amount_line}\n"
    return text


def take_profit_rules(take_profit: float) -> str:
    return (
        rule_block(
            "take_profit_up_position",
            [("has_up_position", "is", "true"), ("up_price", ">=", clean_number(take_profit))],
            "sell_up",
            "",
        )
        + "\n"
        + rule_block(
            "take_profit_down_position",
            [("has_down_position", "is", "true"), ("down_price", ">=", clean_number(take_profit))],
            "sell_down",
            "",
        )
    )


def buy_amount_line(size_mode: str, amount: float) -> str:
    if size_mode == "token":
        return f"token_amount: {int(round(amount))}"
    return f"balance_pct: {clean_number(amount)}"


def sized_amount(amount: float, multiplier: float, size_mode: str) -> float:
    if size_mode == "token":
        return max(1.0, round(amount * multiplier))
    return min(0.95, amount * multiplier)


def cheap_rules(cheap_price: float, size_mode: str, amount: float, down_multiplier: float) -> str:
    down_amount = sized_amount(amount, down_multiplier, size_mode)
    return (
        rule_block(
            "cheap_up_optional_lottery",
            [("up_price", "<=", clean_number(cheap_price))],
            "buy_up",
            buy_amount_line(size_mode, amount),
        )
        + "\n"
        + rule_block(
            "cheap_down_down_bias_lottery",
            [("down_price", "<=", clean_number(cheap_price))],
            "buy_down",
            buy_amount_line(size_mode, down_amount),
        )
    )


def btc_down_bias_rules(entry_cap: float, size_mode: str, amount: float, down_multiplier: float) -> str:
    down_amount = sized_amount(amount, down_multiplier, size_mode)
    return rule_block(
        "btc_below_price_to_beat_buy_down",
        [("btc_below_price_to_beat", "is", "true"), ("down_price", "<=", clean_number(entry_cap))],
        "buy_down",
        buy_amount_line(size_mode, down_amount),
    )


def btc_directional_rules(entry_cap: float, size_mode: str, amount: float, down_multiplier: float) -> str:
    down_amount = sized_amount(amount, down_multiplier, size_mode)
    return (
        rule_block(
            "btc_above_price_to_beat_buy_up",
            [("btc_above_price_to_beat", "is", "true"), ("up_price", "<=", clean_number(entry_cap))],
            "buy_up",
            buy_amount_line(size_mode, amount),
        )
        + "\n"
        + rule_block(
            "btc_below_price_to_beat_buy_down",
            [("btc_below_price_to_beat", "is", "true"), ("down_price", "<=", clean_number(entry_cap))],
            "buy_down",
            buy_amount_line(size_mode, down_amount),
        )
    )


def chaser_rules(momentum_pct: float, entry_cap: float, size_mode: str, amount: float, down_multiplier: float) -> str:
    down_amount = sized_amount(amount, down_multiplier, size_mode)
    return (
        rule_block(
            "up_price_momentum_chase_buy_up",
            [("up_price_pct_change", ">=", clean_number(momentum_pct)), ("up_price", "<=", clean_number(entry_cap))],
            "buy_up",
            buy_amount_line(size_mode, amount),
        )
        + "\n"
        + rule_block(
            "down_price_momentum_chase_buy_down",
            [("down_price_pct_change", ">=", clean_number(momentum_pct)), ("down_price", "<=", clean_number(entry_cap))],
            "buy_down",
            buy_amount_line(size_mode, down_amount),
        )
    )


def fader_rules(spike_pct: float, size_mode: str, amount: float, down_multiplier: float) -> str:
    down_amount = sized_amount(amount, down_multiplier, size_mode)
    return (
        rule_block(
            "fade_up_price_spike_buy_down",
            [("up_price_pct_change", ">=", clean_number(spike_pct)), ("up_price", ">=", 0.60)],
            "buy_down",
            buy_amount_line(size_mode, down_amount),
        )
        + "\n"
        + rule_block(
            "fade_down_price_spike_buy_up",
            [("down_price_pct_change", ">=", clean_number(spike_pct)), ("down_price", ">=", 0.60)],
            "buy_up",
            buy_amount_line(size_mode, amount),
        )
    )


def build_strategy_name(branch_name: str, take_profit: float, cheap_price: float, size_mode: str, amount: float) -> str:
    base = (
        f"{branch_name}"
        f"_tp{clean_number(take_profit).replace('.', '')}"
        f"_cheap{clean_number(cheap_price).replace('.', '')}"
    )
    if size_mode == "token":
        return f"{base}_tok{int(round(amount))}"
    return f"{base}_balpct{clean_pct_code(amount)}"


def build_strategy_yaml(branch: dict, take_profit: float, cheap_price: float, size_mode: str, amount: float) -> str:
    name = build_strategy_name(branch["name"], take_profit, cheap_price, size_mode, amount)
    sections = []
    if "take_profit" in branch["rules"]:
        sections.append(take_profit_rules(take_profit))
    if "cheap" in branch["rules"]:
        sections.append(cheap_rules(cheap_price, size_mode, amount, branch["down_size_multiplier"]))
    if "btc_down_bias" in branch["rules"]:
        sections.append(btc_down_bias_rules(branch["entry_cap"], size_mode, amount, branch["down_size_multiplier"]))
    if "btc_directional" in branch["rules"]:
        sections.append(btc_directional_rules(branch["entry_cap"], size_mode, amount, branch["down_size_multiplier"]))
    if "chaser" in branch["rules"]:
        sections.append(chaser_rules(branch["momentum_pct"], branch["entry_cap"], size_mode, amount, branch["down_size_multiplier"]))
    if "fader" in branch["rules"]:
        sections.append(fader_rules(branch["spike_pct"], size_mode, amount, branch["down_size_multiplier"]))

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
        "batch_id: deterministic_diverse_sizing_suite",
        "markets_folder: simulator_ready_markets",
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
                for token_amount in TOKEN_AMOUNT_VALUES:
                    yaml_text = build_strategy_yaml(branch, take_profit, cheap_price, "token", token_amount)
                    name = yaml_text.splitlines()[0].split(": ", 1)[1]
                    path = OUTPUT_DIR / f"{name}.yaml"
                    path.write_text(yaml_text, encoding="utf-8")
                    strategy_paths.append(path)
                for balance_pct in BALANCE_PCT_VALUES:
                    yaml_text = build_strategy_yaml(branch, take_profit, cheap_price, "balance_pct", balance_pct)
                    name = yaml_text.splitlines()[0].split(": ", 1)[1]
                    path = OUTPUT_DIR / f"{name}.yaml"
                    path.write_text(yaml_text, encoding="utf-8")
                    strategy_paths.append(path)

    BATCH_CONFIG.write_text(build_batch_yaml(strategy_paths), encoding="utf-8")
    print(f"Wrote {len(strategy_paths)} strategy configs to {OUTPUT_DIR}")
    print(f"Wrote batch config to {BATCH_CONFIG}")


if __name__ == "__main__":
    main()
