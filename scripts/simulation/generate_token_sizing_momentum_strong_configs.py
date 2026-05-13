from pathlib import Path


OUTPUT_DIR = Path("configs/strategies/token_sizing_momentum_strong")
BATCH_CONFIG = Path("configs/simulation_token_sizing_momentum_strong.yaml")

TAKE_PROFIT = 0.95
MAX_ORDERS = 300
STARTING_BALANCE = 400
DIRECTIONAL_CAP = 0.85

# This is the cheap ladder from the high-performing token_sizing_momentum run.
CHEAP_TOKEN_LADDER = [
    (0.005, 60),
    (0.015, 40),
    (0.025, 20),
]

TOKEN_DECREASE_SPECS = [
    {"name": "step05_d5", "step": 0.05, "delta_tokens": 5},
    {"name": "step05_d10", "step": 0.05, "delta_tokens": 10},
    {"name": "step05_d15", "step": 0.05, "delta_tokens": 15},
    {"name": "step10_d5", "step": 0.10, "delta_tokens": 5},
    {"name": "step10_d10", "step": 0.10, "delta_tokens": 10},
    {"name": "step10_d15", "step": 0.10, "delta_tokens": 15},
]
TOKEN_INCREASE_MULTIPLIERS = [50, 75, 100]
MOMENTUM_PCTS = [2.5, 5.0, 10.0]
MOMENTUM_TICKS = [1, 2, 3]


def clean_number(value: float) -> str:
    return f"{value:.4f}".rstrip("0").rstrip(".")


def clean_code(value: float) -> str:
    return clean_number(value).replace(".", "")


def yaml_condition(metric: str, operator: str, value, indent: int = 8) -> str:
    pad = " " * indent
    return (
        f"{pad}- metric: {metric}\n"
        f"{pad}  operator: \"{operator}\"\n"
        f"{pad}  value: {value}\n"
    )


def rule_block(name: str, conditions: list[tuple[str, str, object]], action: str, token_amount: int | None = None) -> str:
    text = f"    - name: {name}\n      all:\n"
    for metric, operator, value in conditions:
        text += yaml_condition(metric, operator, value)
    text += f"      action: {action}\n"
    if token_amount is not None:
        text += f"      token_amount: {token_amount}\n"
    return text


def take_profit_rules() -> list[str]:
    return [
        rule_block("take_profit_up_position", [("has_up_position", "is", "true"), ("up_price", ">=", TAKE_PROFIT)], "sell_up"),
        rule_block("take_profit_down_position", [("has_down_position", "is", "true"), ("down_price", ">=", TAKE_PROFIT)], "sell_down"),
    ]


def cheap_rules() -> list[str]:
    rules = []
    for threshold, tokens in CHEAP_TOKEN_LADDER:
        code = clean_code(threshold)
        rules.append(rule_block(f"cheap_up_{code}_buy_{tokens}_tokens", [("up_price", "<=", clean_number(threshold))], "buy_up", tokens))
        rules.append(rule_block(f"cheap_down_{code}_buy_{tokens}_tokens", [("down_price", "<=", clean_number(threshold))], "buy_down", tokens))
    return rules


def price_bands(step: float) -> list[float]:
    values = []
    current = DIRECTIONAL_CAP
    while current >= 0.05:
        values.append(round(current, 2))
        current -= step
    return values


def token_decrease_amount(price: float, step: float, delta_tokens: int) -> int:
    floor = min(price_bands(step))
    levels_above_floor = round((price - floor) / step)
    return max(1, 20 + levels_above_floor * delta_tokens)


def token_increase_amount(price: float, multiplier: int) -> int:
    return max(1, round((1.0 - price) * multiplier))


def directional_rules(strategy_kind: str, step: float, *, delta_tokens: int = 0, multiplier: int = 100) -> list[str]:
    rules = []
    for price in sorted(price_bands(step), reverse=True):
        if strategy_kind == "token_decrease":
            tokens = token_decrease_amount(price, step, delta_tokens)
        elif strategy_kind == "token_increase":
            tokens = token_increase_amount(price, multiplier)
        else:
            raise ValueError(strategy_kind)

        code = clean_code(price)
        rules.append(rule_block(
            f"btc_above_ptb_up_le_{code}_buy_{tokens}_tokens",
            [("btc_above_price_to_beat", "is", "true"), ("up_price", "<=", clean_number(price))],
            "buy_up",
            tokens,
        ))
        rules.append(rule_block(
            f"btc_below_ptb_down_le_{code}_buy_{tokens}_tokens",
            [("btc_below_price_to_beat", "is", "true"), ("down_price", "<=", clean_number(price))],
            "buy_down",
            tokens,
        ))
    return rules


def momentum_metric(side: str, ticks_back: int) -> str:
    if ticks_back == 1:
        return f"{side}_price_pct_change"
    return f"{side}_price_pct_change_{ticks_back}_ticks"


def momentum_rules(momentum_pct: float, ticks_back: int) -> list[str]:
    pct_code = clean_code(momentum_pct)
    return [
        rule_block(
            f"up_price_momentum_{pct_code}_over_{ticks_back}_ticks_buy_up",
            [(momentum_metric("up", ticks_back), ">=", clean_number(momentum_pct)), ("up_price", "<=", DIRECTIONAL_CAP)],
            "buy_up",
            20,
        ),
        rule_block(
            f"down_price_momentum_{pct_code}_over_{ticks_back}_ticks_buy_down",
            [(momentum_metric("down", ticks_back), ">=", clean_number(momentum_pct)), ("down_price", "<=", DIRECTIONAL_CAP)],
            "buy_down",
            20,
        ),
        rule_block(
            f"fade_up_price_spike_{pct_code}_over_{ticks_back}_ticks_buy_down",
            [(momentum_metric("up", ticks_back), ">=", clean_number(momentum_pct)), ("up_price", ">=", 0.60)],
            "buy_down",
            20,
        ),
        rule_block(
            f"fade_down_price_spike_{pct_code}_over_{ticks_back}_ticks_buy_up",
            [(momentum_metric("down", ticks_back), ">=", clean_number(momentum_pct)), ("down_price", ">=", 0.60)],
            "buy_up",
            20,
        ),
    ]


def build_yaml(name: str, rules: list[str]) -> str:
    return (
        f"name: {name}\n"
        "type: rule_based\n"
        "order_usd: 1\n\n"
        "params:\n"
        "  default_usd_amount: 1\n"
        f"  max_orders: {MAX_ORDERS}\n"
        "  cooldown_ticks: 1\n\n"
        "  rules:\n"
        + "\n".join(rules)
    )


def build_batch_yaml(paths: list[Path]) -> str:
    lines = [
        "batch_id: token_sizing_momentum_strong",
        "markets_folder: simulator_ready_markets",
        "market_pattern: btc-updown-5m-*.csv",
        "output_root: runs",
        "max_markets:",
        "",
        f"starting_balance: {STARTING_BALANCE}",
        "order_usd: 1",
        "final_outcome:",
        "compound_balance: false",
        "",
        "strategies:",
    ]
    for path in paths:
        label = path.stem
        lines.extend([
            f"  - config: {path.as_posix()}",
            "    count: 1",
            f"    label: {label}",
            "",
        ])
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    strategy_paths = []

    for spec in TOKEN_DECREASE_SPECS:
        name = f"token_decrease_{spec['name']}"
        rules = take_profit_rules() + cheap_rules() + directional_rules(
            "token_decrease",
            spec["step"],
            delta_tokens=spec["delta_tokens"],
        )
        path = OUTPUT_DIR / f"{name}.yaml"
        path.write_text(build_yaml(name, rules), encoding="utf-8")
        strategy_paths.append(path)

    for step in [0.05, 0.10]:
        for multiplier in TOKEN_INCREASE_MULTIPLIERS:
            name = f"token_increase_step{clean_code(step)}_m{multiplier}"
            rules = take_profit_rules() + cheap_rules() + directional_rules(
                "token_increase",
                step,
                multiplier=multiplier,
            )
            path = OUTPUT_DIR / f"{name}.yaml"
            path.write_text(build_yaml(name, rules), encoding="utf-8")
            strategy_paths.append(path)

    for momentum_pct in MOMENTUM_PCTS:
        for ticks_back in MOMENTUM_TICKS:
            name = f"momentum_swing_{clean_code(momentum_pct)}pct_{ticks_back}ticks"
            rules = take_profit_rules() + momentum_rules(momentum_pct, ticks_back)
            path = OUTPUT_DIR / f"{name}.yaml"
            path.write_text(build_yaml(name, rules), encoding="utf-8")
            strategy_paths.append(path)

    BATCH_CONFIG.write_text(build_batch_yaml(strategy_paths), encoding="utf-8")
    print(f"Wrote {len(strategy_paths)} strategies to {OUTPUT_DIR}")
    print(f"Wrote batch config to {BATCH_CONFIG}")


if __name__ == "__main__":
    main()
