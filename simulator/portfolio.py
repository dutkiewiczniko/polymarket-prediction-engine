\
from dataclasses import dataclass


@dataclass
class Portfolio:
    """Simple Polymarket-style portfolio.

    cash: dollars not currently spent
    up_tokens: number of UP outcome tokens held
    down_tokens: number of DOWN outcome tokens held
    """

    cash: float = 100.0
    up_tokens: float = 0.0
    down_tokens: float = 0.0

    def mark_to_market(self, up_price: float, down_price: float) -> float:
        return self.cash + self.up_tokens * up_price + self.down_tokens * down_price

    def resolve(self, final_outcome: str) -> float:
        outcome = final_outcome.lower().strip()
        if outcome == "up":
            return self.cash + self.up_tokens
        if outcome == "down":
            return self.cash + self.down_tokens
        raise ValueError(f"final_outcome must be 'up' or 'down', got {final_outcome!r}")
