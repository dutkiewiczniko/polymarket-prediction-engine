from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.simulation import generate_token_sizing_momentum_strong_configs as strong


OUTPUT_DIR = Path("configs/strategies/token_decrease_cheap_tail_comparison")
BATCH_CONFIG = Path("configs/simulation_token_decrease_cheap_tail_comparison.yaml")

# Keep this intentionally small: strongest old token-decrease baselines plus
# equivalent versions with a 5c -> 1c cheap ladder.
SPECS = [
    {"name": "step05_d10", "step": 0.05, "delta_tokens": 10},
    {"name": "step05_d15", "step": 0.05, "delta_tokens": 15},
    {"name": "step10_d15", "step": 0.10, "delta_tokens": 15},
]

OLD_CHEAP_LADDER = [
    (0.005, 60),
    (0.015, 40),
    (0.025, 20),
]

CHEAP_5_TO_1_LADDER = [
    (0.01, 100),
    (0.02, 80),
    (0.03, 60),
    (0.04, 40),
    (0.05, 20),
]


def cheap_rules(ladder: list[tuple[float, int]]) -> list[str]:
    rules = []
    for threshold, tokens in ladder:
        code = strong.clean_code(threshold)
        rules.append(
            strong.rule_block(
                f"cheap_up_{code}_buy_{tokens}_tokens",
                [("up_price", "<=", strong.clean_number(threshold))],
                "buy_up",
                tokens,
            )
        )
        rules.append(
            strong.rule_block(
                f"cheap_down_{code}_buy_{tokens}_tokens",
                [("down_price", "<=", strong.clean_number(threshold))],
                "buy_down",
                tokens,
            )
        )
    return rules


def build_strategy(name: str, spec: dict, ladder: list[tuple[float, int]]) -> str:
    rules = (
        strong.take_profit_rules()
        + cheap_rules(ladder)
        + strong.directional_rules(
            "token_decrease",
            spec["step"],
            delta_tokens=spec["delta_tokens"],
        )
    )
    return strong.build_yaml(name, rules)


def build_batch_yaml(paths: list[Path]) -> str:
    lines = [
        "batch_id: token_decrease_cheap_tail_comparison",
        "markets_folder: simulator_ready_markets",
        "market_pattern: btc-updown-5m-*.csv",
        "output_root: runs",
        "max_markets:",
        "",
        f"starting_balance: {strong.STARTING_BALANCE}",
        "order_usd: 1",
        "final_outcome:",
        "compound_balance: false",
        "",
        "strategies:",
    ]
    for path in paths:
        lines.extend(
            [
                f"  - config: {path.as_posix()}",
                "    count: 1",
                f"    label: {path.stem}",
                "",
            ]
        )
    return "\n".join(lines)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = []

    for spec in SPECS:
        for suffix, ladder in [
            ("old_cheap", OLD_CHEAP_LADDER),
            ("cheap_5to1", CHEAP_5_TO_1_LADDER),
        ]:
            name = f"token_decrease_{spec['name']}_{suffix}"
            path = OUTPUT_DIR / f"{name}.yaml"
            path.write_text(build_strategy(name, spec, ladder), encoding="utf-8")
            paths.append(path)

    BATCH_CONFIG.write_text(build_batch_yaml(paths), encoding="utf-8")
    print(f"Wrote {len(paths)} strategies to {OUTPUT_DIR}")
    print(f"Wrote batch config to {BATCH_CONFIG}")


if __name__ == "__main__":
    main()
