from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml


SOURCE = Path("configs/strategies/token_decrease_balance_band_scaled/token_decrease_step10_d15_cheap_5to1_band_scaled.yaml")
OUT_DIR = Path("configs/strategies/token_pricing_curve_comparison")
BATCH_CONFIG = Path("configs/simulation_token_pricing_curve_comparison.yaml")

THRESHOLDS = [0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05]

CURVES = {
    "original_step10_d15": {
        0.85: 140,
        0.75: 125,
        0.65: 110,
        0.55: 95,
        0.45: 80,
        0.35: 65,
        0.25: 50,
        0.15: 35,
        0.05: 20,
    },
    # Requested curve, linearly filled between 0.05->140, 0.30->90, 0.65->110, 0.85->140.
    "u_curve_cheap140_mid90_065110_085140": {
        0.85: 140,
        0.75: 125,
        0.65: 110,
        0.55: 104,
        0.45: 99,
        0.35: 93,
        0.25: 100,
        0.15: 120,
        0.05: 140,
    },
    # Conservative middle-heavy variant: avoids very large cheap-tail directional adds.
    "middle_weighted_cheap60_mid115_high135": {
        0.85: 135,
        0.75: 125,
        0.65: 115,
        0.55: 110,
        0.45: 105,
        0.35: 95,
        0.25: 85,
        0.15: 70,
        0.05: 60,
    },
    # Cheap-heavy variant: tests whether leaning into very low directional prices helps.
    "cheap_heavy_cheap180_mid120_high90": {
        0.85: 90,
        0.75: 100,
        0.65: 110,
        0.55: 118,
        0.45: 125,
        0.35: 130,
        0.25: 145,
        0.15: 165,
        0.05: 180,
    },
    # Confidence-heavy variant: does less at low prices, more when the side is already favored.
    "confidence_heavy_cheap30_mid80_high170": {
        0.85: 170,
        0.75: 145,
        0.65: 120,
        0.55: 100,
        0.45: 85,
        0.35: 70,
        0.25: 55,
        0.15: 40,
        0.05: 30,
    },
}


def threshold_from_rule(rule: dict) -> float | None:
    for condition in rule.get("all", []):
        if condition.get("metric") in {"up_price", "down_price"}:
            value = condition.get("value")
            if value in THRESHOLDS:
                return float(value)
    return None


def rewrite_curve(config: dict, name: str, curve: dict[float, int]) -> dict:
    cfg = deepcopy(config)
    cfg["name"] = f"token_pricing_{name}_band_scaled"

    for rule in cfg.get("params", {}).get("rules", []):
        rule_name = str(rule.get("name", ""))
        if not rule_name.startswith(("btc_above_ptb_", "btc_below_ptb_")):
            continue
        threshold = threshold_from_rule(rule)
        if threshold is None:
            continue
        amount = curve[threshold]
        rule["balance_scaled_token_amount"] = amount
        side = "up" if "btc_above_ptb_" in rule_name else "down"
        threshold_label = f"{int(round(threshold * 100)):03d}"
        rule["name"] = f"btc_{'above' if side == 'up' else 'below'}_ptb_{side}_le_{threshold_label}_buy_{amount}_tokens"
    return cfg


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    source_cfg = yaml.safe_load(SOURCE.read_text(encoding="utf-8"))

    strategy_paths = []
    for name, curve in CURVES.items():
        cfg = rewrite_curve(source_cfg, name, curve)
        path = OUT_DIR / f"{cfg['name']}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        strategy_paths.append((cfg["name"], path))

    lines = [
        "batch_id: token_pricing_curve_comparison",
        "markets_folder: simulator_ready_markets",
        "market_pattern: btc-updown-5m-*.csv",
        "output_root: runs",
        "max_markets:",
        "",
        "starting_balance: 1000",
        "order_usd: 1",
        "final_outcome:",
        "compound_balance: true",
        "",
        "effective_market_balance_bands:",
        "  - [31.25, 62.5, 3]",
        "  - [62.5, 125, 6]",
        "  - [125, 250, 12]",
        "  - [250, 500, 25]",
        "  - [500, 1000, 50]",
        "  - [1000, 2000, 100]",
        "  - [2000, 4000, 200]",
        "  - [4000, 8000, 400]",
        "  - [8000, 16000, 800]",
        "  - [16000, 32000, 1600]",
        "  - [32000, 64000, 3200]",
        "  - [64000, 128000, 6400]",
        "  - [128000, 256000, 12800]",
        "  - [256000, 512000, 25600]",
        "  - [512000, 1024000, 51200]",
        "  - [1024000, 2048000, 102400]",
        "",
        "effective_market_balance_default: 0",
        "",
        "strategies:",
    ]
    for label, path in strategy_paths:
        lines.extend([
            f"  - config: {path.as_posix()}",
            "    count: 1",
            f"    label: {label}",
            "",
        ])
    BATCH_CONFIG.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote {len(strategy_paths)} strategies to {OUT_DIR}")
    print(f"Wrote batch config to {BATCH_CONFIG}")


if __name__ == "__main__":
    main()
