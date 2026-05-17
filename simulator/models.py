\
from dataclasses import dataclass, field
from typing import Optional


@dataclass(frozen=True)
class MarketTick:
    """One replayable market state from a recorded CSV row."""
    timestamp: str
    unix_time: float
    seconds_left: Optional[float]
    elapsed: Optional[float]
    up_price: Optional[float]
    down_price: Optional[float]
    btc_binance: Optional[float]
    btc_chainlink: Optional[float]
    price_to_beat: Optional[float]
    extra: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DecisionState:
    """State passed into a strategy at one decision point."""
    tick: MarketTick
    cash: float
    up_tokens: float
    down_tokens: float
    current_balance: float
    market_start_balance: float
    market_spend_used: float
    last_action: str
    orders_placed: int


@dataclass(frozen=True)
class TradeEvent:
    timestamp: str
    action: str
    side: str
    price: float
    usd_amount: float
    tokens: float
    cash_after: float
    up_tokens_after: float
    down_tokens_after: float
    reason: str = ""


@dataclass(frozen=True)
class SimulationResult:
    market_file: str
    strategy_name: str
    final_outcome: str
    final_balance: float
    starting_balance: float
    total_reward: float
    rows_written: int
