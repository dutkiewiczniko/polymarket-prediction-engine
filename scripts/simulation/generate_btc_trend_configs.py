"""Generate the btc_trend_experiment strategy config family.

Macro BTC trend followers: each variant reads the per-market
btc_change_<period>_pct columns (added by scripts/data/enrich_markets_with_btc_trend.py)
and, when the trend into market start is strong enough, trades the side BTC is
moving towards:

- open entry: small buy while the favored side is still near 50c early on
- cheap adds: larger buys whenever the favored side dips cheap mid-market
- take profit: sell any position at 0.95

"Strong" thresholds come from scripts/analysis/analyze_btc_trend_predictiveness.py
(median |change| per period on the 320-market bench):
1h ~0.2, 12h ~1.0, 24h ~1.5, 7d ~5.0.

The struct_down_unconditional control has the identical structure/sizing but
always favors Down with no trend condition -- on a down-regime window it
isolates whether the trend filter adds anything over plain down bias.

Usage:
    python scripts/simulation/generate_btc_trend_configs.py
"""

import sys
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

OUTPUT_DIR = REPO_ROOT / "configs" / "strategies" / "btc_trend_experiment"

OPEN_ENTRY_BALANCE_PCT = 0.10
CHEAP_ADD_BALANCE_PCT = 0.08
OPEN_POOL_PCT = 0.10
CHEAP_POOL_PCT = 0.40
OPEN_PRICE_MIN = 0.45
OPEN_PRICE_MAX = 0.58
OPEN_MAX_ELAPSED_S = 90
CHEAP_MAX_PRICE = 0.35
CHEAP_MIN_SECONDS_LEFT = 45
CHEAP_ADD_COOLDOWN_TICKS = 40
TAKE_PROFIT_PRICE = 0.95

# period label -> "strong" threshold (pct) from the calibration analysis
STRONG_THRESHOLDS = {"1h": 0.2, "12h": 1.0, "24h": 1.5, "7d": 5.0}

# variant name -> (list of period labels that must all agree, include_open_entry)
VARIANTS = {
    "trend_1h": (["1h"], True),
    "trend_12h": (["12h"], True),
    "trend_24h": (["24h"], True),
    "trend_7d": (["7d"], True),
    "trend_12h_24h_confirm": (["12h", "24h"], True),
    "trend_24h_cheap_only": (["24h"], False),
}

# Anti (fade) variants: same structure, but buy the side OPPOSITE the trend
# (strong downtrend -> buy up). Motivated by the calibration's mean-reversion
# hint at trend extremes.
ANTI_VARIANTS = {
    "anti_trend_12h": (["12h"], True),
    "anti_trend_24h": (["24h"], True),
    "anti_trend_7d": (["7d"], True),
}


def take_profit_rules() -> list[dict]:
    return [
        {
            "name": f"take_profit_{side}_position",
            "all": [
                {"metric": f"has_{side}_position", "operator": "is", "value": True},
                {"metric": f"{side}_price", "operator": ">=", "value": TAKE_PROFIT_PRICE},
            ],
            "action": f"sell_{side}",
        }
        for side in ("up", "down")
    ]


def trend_conditions(periods: list[str], side: str) -> list[dict]:
    operator = ">=" if side == "up" else "<="
    sign = 1.0 if side == "up" else -1.0
    return [
        {
            "metric": f"btc_change_{period}_pct",
            "operator": operator,
            "value": sign * STRONG_THRESHOLDS[period],
        }
        for period in periods
    ]


def signal_side_for(side: str, anti: bool) -> str:
    if not anti:
        return side
    return "down" if side == "up" else "up"


def open_entry_rule(periods: list[str], side: str, anti: bool = False) -> dict:
    return {
        "name": f"{'anti_' if anti else ''}trend_open_entry_{side}",
        "pool": "open_entry",
        "all": [
            *trend_conditions(periods, signal_side_for(side, anti)),
            {"metric": f"{side}_price", "operator": ">=", "value": OPEN_PRICE_MIN},
            {"metric": f"{side}_price", "operator": "<=", "value": OPEN_PRICE_MAX},
            {"metric": "elapsed", "operator": "<=", "value": OPEN_MAX_ELAPSED_S},
        ],
        "action": f"buy_{side}",
        "balance_pct": OPEN_ENTRY_BALANCE_PCT,
    }


def cheap_add_rule(periods: list[str], side: str, anti: bool = False) -> dict:
    return {
        "name": f"{'anti_' if anti else ''}trend_cheap_add_{side}",
        "pool": "cheap_adds",
        "cooldown_ticks": CHEAP_ADD_COOLDOWN_TICKS,
        "all": [
            *trend_conditions(periods, signal_side_for(side, anti)),
            {"metric": f"{side}_price", "operator": "<=", "value": CHEAP_MAX_PRICE},
            {"metric": "seconds_left", "operator": ">=", "value": CHEAP_MIN_SECONDS_LEFT},
        ],
        "action": f"buy_{side}",
        "balance_pct": CHEAP_ADD_BALANCE_PCT,
    }


def build_config(name: str, periods: list[str] | None, include_open_entry: bool, anti: bool = False) -> dict:
    """periods=None -> unconditional down-only control (no trend conditions)."""
    rules = take_profit_rules()
    sides = ("up", "down") if periods is not None else ("down",)
    for side in sides:
        if include_open_entry:
            rule = open_entry_rule(periods or [], side, anti=anti)
            if periods is None:
                rule["name"] = f"struct_open_entry_{side}"
            rules.append(rule)
    for side in sides:
        rule = cheap_add_rule(periods or [], side, anti=anti)
        if periods is None:
            rule["name"] = f"struct_cheap_add_{side}"
        rules.append(rule)

    return {
        "name": f"btc_trend_{name}" if periods is not None else name,
        "type": "rule_based",
        "starting_balance": 100,
        "order_usd": 1,
        "params": {
            "default_usd_amount": 1,
            "max_orders": 200,
            "cooldown_ticks": 3,
            "cooldown_scope": "per_rule",
            "pools": {
                "open_entry": {"max_pool_spend_pct": OPEN_POOL_PCT},
                "cheap_adds": {"max_pool_spend_pct": CHEAP_POOL_PCT},
            },
            "rules": rules,
        },
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    configs = {"struct_down_unconditional": build_config("struct_down_unconditional", None, True)}
    for name, (periods, include_open_entry) in VARIANTS.items():
        configs[name] = build_config(name, periods, include_open_entry)
    for name, (periods, include_open_entry) in ANTI_VARIANTS.items():
        configs[name] = build_config(name, periods, include_open_entry, anti=True)

    for name, config in configs.items():
        path = OUTPUT_DIR / f"{name}.yaml"
        with path.open("w", encoding="utf-8") as f:
            f.write(f"# Auto-generated by {Path(__file__).name}\n")
            yaml.safe_dump(config, f, sort_keys=False, default_flow_style=False)
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
