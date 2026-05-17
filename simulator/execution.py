\
from simulator.models import TradeEvent
from simulator.portfolio import Portfolio


def execute_action(
    *,
    portfolio: Portfolio,
    action: str,
    timestamp: str,
    up_price: float,
    down_price: float,
    usd_amount: float,
    sell_opposite_first: bool = True,
    max_buy_usd: float | None = None,
    max_sell_tokens: float | None = None,
    reason: str = "",
) -> list[TradeEvent]:
    """Apply one action to the portfolio.

    Supported actions:
        hold
        buy_up
        buy_down
        sell_up
        sell_down

    This is intentionally simple at first:
    - market orders fill immediately
    - no fees
    - optional liquidity caps can produce partial fills

    Those can be added later once replay works.
    """

    events: list[TradeEvent] = []
    action = action.lower().strip()

    if action == "hold":
        return events

    if action == "buy_up":
        if sell_opposite_first and portfolio.down_tokens > 0:
            events.extend(_sell_down(portfolio, timestamp, down_price, reason="exit down before buy_up", max_tokens=max_sell_tokens))
        if max_buy_usd is not None:
            usd_amount = min(usd_amount, max(0.0, max_buy_usd))
        events.extend(_buy_up(portfolio, timestamp, up_price, usd_amount, reason))
        return events

    if action == "buy_down":
        if sell_opposite_first and portfolio.up_tokens > 0:
            events.extend(_sell_up(portfolio, timestamp, up_price, reason="exit up before buy_down", max_tokens=max_sell_tokens))
        if max_buy_usd is not None:
            usd_amount = min(usd_amount, max(0.0, max_buy_usd))
        events.extend(_buy_down(portfolio, timestamp, down_price, usd_amount, reason))
        return events

    if action == "sell_up":
        events.extend(_sell_up(portfolio, timestamp, up_price, reason, max_tokens=max_sell_tokens))
        return events

    if action == "sell_down":
        events.extend(_sell_down(portfolio, timestamp, down_price, reason, max_tokens=max_sell_tokens))
        return events

    raise ValueError(f"Unknown action: {action!r}")


def _buy_up(portfolio: Portfolio, timestamp: str, price: float, usd_amount: float, reason: str) -> list[TradeEvent]:
    if price <= 0 or usd_amount <= 0 or portfolio.cash < usd_amount:
        return []

    tokens = usd_amount / price
    portfolio.cash -= usd_amount
    portfolio.up_tokens += tokens

    return [TradeEvent(
        timestamp=timestamp,
        action="buy",
        side="up",
        price=price,
        usd_amount=usd_amount,
        tokens=tokens,
        cash_after=portfolio.cash,
        up_tokens_after=portfolio.up_tokens,
        down_tokens_after=portfolio.down_tokens,
        reason=reason,
    )]


def _buy_down(portfolio: Portfolio, timestamp: str, price: float, usd_amount: float, reason: str) -> list[TradeEvent]:
    if price <= 0 or usd_amount <= 0 or portfolio.cash < usd_amount:
        return []

    tokens = usd_amount / price
    portfolio.cash -= usd_amount
    portfolio.down_tokens += tokens

    return [TradeEvent(
        timestamp=timestamp,
        action="buy",
        side="down",
        price=price,
        usd_amount=usd_amount,
        tokens=tokens,
        cash_after=portfolio.cash,
        up_tokens_after=portfolio.up_tokens,
        down_tokens_after=portfolio.down_tokens,
        reason=reason,
    )]


def _sell_up(portfolio: Portfolio, timestamp: str, price: float, reason: str, max_tokens: float | None = None) -> list[TradeEvent]:
    if price <= 0 or portfolio.up_tokens <= 0:
        return []

    tokens = portfolio.up_tokens
    if max_tokens is not None:
        tokens = min(tokens, max(0.0, max_tokens))
    if tokens <= 0:
        return []
    revenue = tokens * price
    portfolio.cash += revenue
    portfolio.up_tokens -= tokens

    return [TradeEvent(
        timestamp=timestamp,
        action="sell",
        side="up",
        price=price,
        usd_amount=revenue,
        tokens=tokens,
        cash_after=portfolio.cash,
        up_tokens_after=portfolio.up_tokens,
        down_tokens_after=portfolio.down_tokens,
        reason=reason,
    )]


def _sell_down(portfolio: Portfolio, timestamp: str, price: float, reason: str, max_tokens: float | None = None) -> list[TradeEvent]:
    if price <= 0 or portfolio.down_tokens <= 0:
        return []

    tokens = portfolio.down_tokens
    if max_tokens is not None:
        tokens = min(tokens, max(0.0, max_tokens))
    if tokens <= 0:
        return []
    revenue = tokens * price
    portfolio.cash += revenue
    portfolio.down_tokens -= tokens

    return [TradeEvent(
        timestamp=timestamp,
        action="sell",
        side="down",
        price=price,
        usd_amount=revenue,
        tokens=tokens,
        cash_after=portfolio.cash,
        up_tokens_after=portfolio.up_tokens,
        down_tokens_after=portfolio.down_tokens,
        reason=reason,
    )]
