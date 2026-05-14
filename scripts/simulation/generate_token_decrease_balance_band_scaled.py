from pathlib import Path
import re


SOURCE_DIR = Path("configs/strategies/token_decrease_cheap_tail_comparison")
OUTPUT_DIR = Path("configs/strategies/token_decrease_balance_band_scaled")
BATCH_CONFIG = Path("configs/simulation_token_decrease_balance_band_scaled.yaml")

SOURCE_FILES = [
    "token_decrease_step05_d10_cheap_5to1.yaml",
    "token_decrease_step05_d15_cheap_5to1.yaml",
    "token_decrease_step10_d15_cheap_5to1.yaml",
]

STARTING_BALANCE = 1000
MIN_EFFECTIVE_BALANCE = 3
MAX_EFFECTIVE_BALANCE = 102400
BASE_BALANCE_BAND = 1000
BASE_EFFECTIVE_BALANCE = 100


def convert_yaml(text: str) -> str:
    text = re.sub(
        r"^name:\s*(.+)$",
        lambda match: f"name: {match.group(1)}_band_scaled",
        text,
        count=1,
        flags=re.MULTILINE,
    )
    return re.sub(r"^(\s*)token_amount:", r"\1balance_scaled_token_amount:", text, flags=re.MULTILINE)


def balance_bands() -> list[tuple[float, float, int]]:
    bands = []

    lower = float(BASE_BALANCE_BAND)
    effective = BASE_EFFECTIVE_BALANCE
    while effective >= MIN_EFFECTIVE_BALANCE:
        bands.append((lower, lower * 2, effective))
        lower /= 2
        effective //= 2

    lower = float(BASE_BALANCE_BAND * 2)
    effective = BASE_EFFECTIVE_BALANCE * 2
    while effective <= MAX_EFFECTIVE_BALANCE:
        upper = lower * 2
        bands.append((lower, upper, effective))
        lower = upper
        effective *= 2

    bands.sort(key=lambda band: band[0])
    return bands


def clean_band_number(value: float) -> str:
    return str(int(value)) if value.is_integer() else f"{value:.4f}".rstrip("0").rstrip(".")


def build_batch_yaml(paths: list[Path]) -> str:
    lines = [
        "batch_id: token_decrease_balance_band_scaled",
        "markets_folder: simulator_ready_markets",
        "market_pattern: btc-updown-5m-*.csv",
        "output_root: runs",
        "max_markets:",
        "",
        "# This is the real/master balance that compounds across markets.",
        f"starting_balance: {STARTING_BALANCE}",
        "order_usd: 1",
        "final_outcome:",
        "compound_balance: true",
        "",
        "# [min_total_balance, max_total_balance, effective_market_balance]",
        "# Order sizes use balance_scaled_token_amount * effective_market_balance / 100.",
        "effective_market_balance_bands:",
    ]
    for min_balance, max_balance, effective_balance in balance_bands():
        lines.append(
            f"  - [{clean_band_number(min_balance)}, {clean_band_number(max_balance)}, {effective_balance}]"
        )
    lines.extend([
        "",
        "# If balance is outside the bands, do not place market exposure.",
        "effective_market_balance_default: 0",
        "",
        "strategies:",
    ])
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
    for filename in SOURCE_FILES:
        source_path = SOURCE_DIR / filename
        output_path = OUTPUT_DIR / filename.replace(".yaml", "_band_scaled.yaml")
        output_path.write_text(convert_yaml(source_path.read_text(encoding="utf-8")), encoding="utf-8")
        paths.append(output_path)

    BATCH_CONFIG.write_text(build_batch_yaml(paths), encoding="utf-8")
    print(f"Wrote {len(paths)} strategies to {OUTPUT_DIR}")
    print(f"Wrote batch config to {BATCH_CONFIG}")


if __name__ == "__main__":
    main()
