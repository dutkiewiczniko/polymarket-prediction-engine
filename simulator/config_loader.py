from pathlib import Path
import copy
import yaml

from simulator.strategies import HoldStrategy, RandomStrategy, MomentumStrategy, RuleBasedStrategy, BaseStrategy


def build_strategy_from_config(cfg: dict) -> BaseStrategy:
    """Build a strategy object from a config dictionary."""

    cfg = cfg or {}
    strategy_type = cfg.get("type", "hold")

    if strategy_type == "hold":
        strategy = HoldStrategy()
        strategy.name = cfg.get("name", strategy.name)
        return strategy

    if strategy_type == "random":
        params = cfg.get("params", {})
        strategy = RandomStrategy(
            hold_probability=float(params.get("hold_probability", 0.70)),
            buy_up_probability=float(params.get("buy_up_probability", 0.15)),
            seed=params.get("seed"),
        )
        strategy.name = cfg.get("name", strategy.name)
        return strategy

    if strategy_type == "momentum_basic":
        params = cfg.get("params", {})
        strategy = MomentumStrategy(
            momentum_pct=float(params.get("momentum_pct", 0.10)),
            min_elapsed_s=float(params.get("min_elapsed_s", 15.0)),
            min_remaining_s=float(params.get("min_remaining_s", 10.0)),
            min_up_prob=float(params.get("min_up_prob", 0.30)),
            max_up_prob=float(params.get("max_up_prob", 0.70)),
        )
        strategy.name = cfg.get("name", strategy.name)
        return strategy

    if strategy_type == "rule_based":
        params = cfg.get("params", {})
        strategy = RuleBasedStrategy(
            rules=params.get("rules", []),
            default_usd_amount=float(params.get("default_usd_amount", cfg.get("order_usd", 1.0))),
            max_orders=int(params["max_orders"]) if params.get("max_orders") is not None else None,
            cooldown_ticks=int(params.get("cooldown_ticks", 0)),
        )
        strategy.name = cfg.get("name", strategy.name)
        return strategy

    raise ValueError(f"Unknown strategy type: {strategy_type!r}")


def load_yaml(path: str | Path) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_strategy_from_yaml(path: str | Path) -> tuple[BaseStrategy, dict]:
    """Load a strategy config from YAML.

    Requires:
        pip install pyyaml
    """

    cfg = load_yaml(path)
    return build_strategy_from_config(cfg), cfg


def clone_strategy_config_with_seed(cfg: dict, seed: int | None) -> dict:
    """Return a copy of a strategy config with the random seed replaced.

    This is mainly useful when running 2x, 10x, or 100x random strategies.
    Without changing the seed, every random run would make the exact same decisions.
    """

    new_cfg = copy.deepcopy(cfg)

    if seed is not None:
        new_cfg.setdefault("params", {})
        new_cfg["params"]["seed"] = seed

    return new_cfg
