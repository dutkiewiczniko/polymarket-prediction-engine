def parse_float(value):
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def liquidity_limits_for_action(
    *,
    action: str,
    row_metrics: dict,
    requested_usd: float,
    cash: float,
    up_tokens: float,
    down_tokens: float,
    depth_window_cents: int = 2,
    fill_fraction: float = 1.0,
    missing_depth_policy: str = "skip",
) -> dict:
    """Translate logged order-book depth into executable buy/sell caps.

    The market CSV stores aggregate visible depth within N cents of best bid/ask.
    This helper uses those columns to cap paper execution. It does not model
    queue priority or exact price walking; it simply prevents larger-than-visible
    fantasy fills.
    """

    action = str(action or "hold").lower().strip()
    fill_fraction = max(0.0, min(1.0, float(fill_fraction)))
    window = int(depth_window_cents)
    result = {
        "requested_usd_amount": requested_usd,
        "executable_usd_amount": requested_usd,
        "max_buy_usd": None,
        "max_sell_tokens": None,
        "reason": "liquidity disabled",
    }

    if action == "hold":
        result["executable_usd_amount"] = 0.0
        result["reason"] = "hold"
        return result

    buy_side = None
    sell_side = None
    if action == "buy_up":
        buy_side = "up"
        sell_side = "down" if down_tokens > 0 else None
    elif action == "buy_down":
        buy_side = "down"
        sell_side = "up" if up_tokens > 0 else None
    elif action == "sell_up":
        sell_side = "up"
    elif action == "sell_down":
        sell_side = "down"

    notes = []
    executable_usd = float(requested_usd)
    max_buy_usd = None
    max_sell_tokens = None

    if buy_side:
        executable_usd = min(executable_usd, float(cash))
        ask_usd = parse_float(row_metrics.get(f"{buy_side}_ask_usd_within_{window}c"))
        if ask_usd is None:
            if missing_depth_policy == "skip":
                executable_usd = 0.0
                notes.append(f"missing {buy_side} ask depth")
            else:
                notes.append(f"missing {buy_side} ask depth, allowed")
        else:
            max_buy_usd = max(0.0, ask_usd * fill_fraction)
            if executable_usd > max_buy_usd:
                notes.append(f"buy capped by {buy_side} ask depth {window}c")
            executable_usd = min(executable_usd, max_buy_usd)

    if sell_side:
        bid_size = parse_float(row_metrics.get(f"{sell_side}_bid_size_within_{window}c"))
        position_tokens = up_tokens if sell_side == "up" else down_tokens
        if bid_size is None:
            if missing_depth_policy == "skip":
                max_sell_tokens = 0.0
                notes.append(f"missing {sell_side} bid depth")
            else:
                notes.append(f"missing {sell_side} bid depth, allowed")
        else:
            max_sell_tokens = min(float(position_tokens), max(0.0, bid_size * fill_fraction))
            if max_sell_tokens < float(position_tokens):
                notes.append(f"sell capped by {sell_side} bid depth {window}c")

    result.update({
        "executable_usd_amount": executable_usd,
        "max_buy_usd": max_buy_usd,
        "max_sell_tokens": max_sell_tokens,
        "reason": "; ".join(notes) if notes else "full requested size available",
    })
    return result
