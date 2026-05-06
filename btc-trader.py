"""
Rule-Based BTC Up/Down Trader (MOCK)
=====================================
Web dashboard replaces matplotlib — open http://localhost:5050

Polls Polymarket Up price fast, listens to Binance + Chainlink,
fires mock BUY UP / BUY DOWN, tracks a mock account, shows
everything on a live browser dashboard with countdown timer.

Data saved:
  data/<slug>.csv          — price ticks per market
  data/trades.csv          — every mock order + P&L after resolution
  data/account_history.csv — balance snapshots

Run:  pip install flask flask-socketio aiohttp websockets requests
      python trader.py
      → open http://localhost:5050
"""

import time
import json
import os
import csv
import asyncio
import threading
from datetime import datetime
from dataclasses import dataclass, asdict

import requests
import aiohttp
import websockets

from flask import Flask, render_template_string
from flask_socketio import SocketIO


# ══════════════════════════════════════════════════════════════════════════════
# TUNABLE PARAMETERS
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class Rules:
    # ── Bet sizing ───────────────────────────────────────────────────────
    base_bet_usd: float = 1.0           # $1 default per bet
    bet_multiplier: int = 1             # actual bet = base × multiplier

    # ── Price momentum ───────────────────────────────────────────────────
    momentum_pct: float = 0.15          # BTC must move this % in window
    momentum_window_s: float = 30.0     # look-back seconds

    # ── Distance to target ───────────────────────────────────────────────
    target_proximity_pct: float = 0.05  # within this % of price-to-beat

    # ── Polymarket odds filter ───────────────────────────────────────────
    min_up_prob: float = 0.30
    max_up_prob: float = 0.70

    # ── Timing ───────────────────────────────────────────────────────────
    min_elapsed_s: float = 15.0
    min_remaining_s: float = 10.0

    # ── Cooldown ─────────────────────────────────────────────────────────
    cooldown_s: float = 30.0

    # ── Mock account ─────────────────────────────────────────────────────
    starting_balance: float = 100.0


RULES = Rules()


# ══════════════════════════════════════════════════════════════════════════════
# SHARED STATE
# ══════════════════════════════════════════════════════════════════════════════

GAMMA_API  = "https://gamma-api.polymarket.com"
CLOB_API   = "https://clob.polymarket.com"
RTDS_URL   = "wss://ws-live-data.polymarket.com"
BINANCE_WS = "wss://stream.binance.com:9443/ws/btcusdt@ticker"
DATA_DIR   = "data_uncleaned"
os.makedirs(DATA_DIR, exist_ok=True)


def daily_data_dir(unix_ts=None):
    """Return the YYYY-MM-DD data folder, creating it when needed."""
    if unix_ts is None:
        unix_ts = time.time()
    day = datetime.fromtimestamp(unix_ts).strftime("%Y-%m-%d")
    path = os.path.join(DATA_DIR, day)
    os.makedirs(path, exist_ok=True)
    return path

live = {
    "btc_binance": [],          # [(unix_ts, price), ...]
    "btc_chainlink": [],
    "up_price": [],
    "down_price": [],

    "price_to_beat": None,
    "market_question": "",
    "current_slug": "",
    "market_start": 0,
    "market_end": 0,

    "markets_seen": 0,

    "running": True,
}
lock = threading.Lock()


# ══════════════════════════════════════════════════════════════════════════════
# CSV WRITERS
# ══════════════════════════════════════════════════════════════════════════════

_tick_file = None
_tick_writer = None
_tick_lock = threading.Lock()  # protects _tick_writer/_tick_file

# Stable market-only schema for recorded live market ticks. Simulated
# portfolio/action/reward fields are added later by the offline simulator.
MARKET_CSV_HEADER = [
    "time_fraction_elapsed",
    "up_price",
    "up_price_change_0_5s",
    "up_price_change_1s",
    "up_price_change_5s",
    "up_price_change_15s",
    "up_price_change_30s",
    "btc_return_0_5s",
    "btc_return_1s",
    "btc_return_5s",
    "btc_return_15s",
    "btc_return_30s",
    "btc_return_60s",
    "btc_volatility_5s",
    "btc_volatility_15s",
    "btc_volatility_30s",
    "btc_volatility_60s",
    "btc_distance_to_beat_pct",
    "market_confidence_gap",
]


def latest_value_at_or_before(series, cutoff_ts):
    value = None
    for ts, price in reversed(series):
        if ts <= cutoff_ts:
            value = price
            break
    return value


def raw_change(series, now_ts, window_s):
    if not series:
        return ""
    current = series[-1][1]
    previous = latest_value_at_or_before(series, now_ts - window_s)
    if current is None or previous is None:
        return ""
    return current - previous


def decimal_return(series, now_ts, window_s):
    if not series:
        return ""
    current = series[-1][1]
    previous = latest_value_at_or_before(series, now_ts - window_s)
    if current is None or previous in (None, 0):
        return ""
    return (current - previous) / previous


def short_return_volatility(series, now_ts, window_s):
    window = [(ts, price) for ts, price in series if ts >= now_ts - window_s and price not in (None, 0)]
    if len(window) < 3:
        return 0.0

    returns = []
    for (_, previous), (_, current) in zip(window, window[1:]):
        if previous:
            returns.append((current - previous) / previous)

    if len(returns) < 2:
        return 0.0

    mean = sum(returns) / len(returns)
    variance = sum((value - mean) ** 2 for value in returns) / len(returns)
    return variance ** 0.5


def format_csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return value


def _market_row_dict():
    """Snapshot current market state and market-only engineered features."""
    now = time.time()

    with lock:
        market_start = live["market_start"]
        market_end = live["market_end"]
        up_series = list(live["up_price"])
        down_series = list(live["down_price"])
        btc_binance_series = list(live["btc_binance"])
        btc_chainlink_series = list(live["btc_chainlink"])
        price_to_beat = live["price_to_beat"]

    seconds_left = (market_end - now) if market_end else ""
    elapsed = (now - market_start) if market_start else ""
    time_fraction_elapsed = (elapsed / 300.0) if isinstance(elapsed, (int, float)) else ""

    up_price = up_series[-1][1] if up_series else None
    down_price = down_series[-1][1] if down_series else None
    btc_binance = btc_binance_series[-1][1] if btc_binance_series else None
    btc_chainlink = btc_chainlink_series[-1][1] if btc_chainlink_series else None
    btc_price = btc_chainlink if btc_chainlink is not None else btc_binance
    btc_series = btc_chainlink_series if btc_chainlink_series else btc_binance_series

    btc_distance_to_beat_pct = ""
    market_confidence_gap = ""

    if btc_price is not None and price_to_beat not in (None, 0):
        btc_distance_to_beat = btc_price - price_to_beat
        btc_distance_to_beat_pct = btc_distance_to_beat / price_to_beat
        btc_above_price_to_beat = 1 if btc_price > price_to_beat else 0
        if up_price is not None:
            market_confidence_gap = btc_above_price_to_beat - up_price

    return {
        "time_fraction_elapsed": time_fraction_elapsed,
        "up_price": up_price,
        "up_price_change_0_5s": raw_change(up_series, now, 0.5),
        "up_price_change_1s": raw_change(up_series, now, 1.0),
        "up_price_change_5s": raw_change(up_series, now, 5.0),
        "up_price_change_15s": raw_change(up_series, now, 15.0),
        "up_price_change_30s": raw_change(up_series, now, 30.0),
        "btc_return_0_5s": decimal_return(btc_series, now, 0.5),
        "btc_return_1s": decimal_return(btc_series, now, 1.0),
        "btc_return_5s": decimal_return(btc_series, now, 5.0),
        "btc_return_15s": decimal_return(btc_series, now, 15.0),
        "btc_return_30s": decimal_return(btc_series, now, 30.0),
        "btc_return_60s": decimal_return(btc_series, now, 60.0),
        "btc_volatility_5s": short_return_volatility(btc_series, now, 5.0),
        "btc_volatility_15s": short_return_volatility(btc_series, now, 15.0),
        "btc_volatility_30s": short_return_volatility(btc_series, now, 30.0),
        "btc_volatility_60s": short_return_volatility(btc_series, now, 60.0),
        "btc_distance_to_beat_pct": btc_distance_to_beat_pct,
        "market_confidence_gap": market_confidence_gap,
    }


def write_market_row():
    """Append one market-only tick row to the active per-market CSV."""
    row = _market_row_dict()

    with _tick_lock:
        if _tick_writer is None or _tick_file is None:
            return  # market not active yet
        _tick_writer.writerow([format_csv_value(row.get(column)) for column in MARKET_CSV_HEADER])
        _tick_file.flush()


def log_balance(event: str):
    return


# ══════════════════════════════════════════════════════════════════════════════
# MOCK ORDER EXECUTION
# ══════════════════════════════════════════════════════════════════════════════

def execute_order(side: str, reason: str):
    return

    """
    Mock order with position awareness.

    How Polymarket tokens work:
    - BUY:  pay $prob per token, deducted from balance
    - SELL: sell tokens you hold at current prob, credited to balance
    - RESOLVE: winning tokens pay $1 each, losing tokens pay $0

    If we hold tokens on the OPPOSITE side, we sell them first (exit),
    then buy on the new side. This avoids holding contradictory positions.
    """
    bet_usd = RULES.base_bet_usd * RULES.bet_multiplier
    now = time.time()

    with lock:
        if live["balance"] < bet_usd and (
            (side == "up" and live["down_tokens"] == 0) or
            (side == "down" and live["up_tokens"] == 0)
        ):
            return  # can't afford and nothing to sell
        btc = live["btc_chainlink"][-1][1] if live["btc_chainlink"] else None
        up_prob = live["up_price"][-1][1] if live["up_price"] else None
        down_prob = live["down_price"][-1][1] if live["down_price"] else None

    if up_prob is None:
        return

    # ── Step 1: Sell opposing position if we have one ─────────────────
    opposite = "down" if side == "up" else "up"
    opp_tokens_key = f"{opposite}_tokens"
    opp_cost_key = f"{opposite}_cost"

    with lock:
        opp_tokens = live[opp_tokens_key]

    if opp_tokens > 0:
        sell_prob = up_prob if opposite == "up" else down_prob
        sell_revenue = opp_tokens * sell_prob  # sell at current market price

        sell_order = {
            "time": now,
            "time_str": datetime.now().strftime("%H:%M:%S"),
            "action": "SELL",
            "side": opposite,
            "tokens": opp_tokens,
            "price": sell_prob,
            "revenue": sell_revenue,
            "bet_usd": 0,
            "buy_prob": sell_prob,
            "btc_price": btc,
            "up_prob": up_prob,
            "reason": f"exit {opposite} → entering {side}",
            "outcome": "sold",
            "payout": sell_revenue,
            "pnl": sell_revenue - live[opp_cost_key],
            "slug": live["current_slug"],
        }

        with lock:
            live["balance"] += sell_revenue
            live[opp_tokens_key] = 0.0
            live[opp_cost_key] = 0.0
            live["orders"].append(sell_order)
            live["all_orders"].append(sell_order)
            live["balance_history"].append((now, live["balance"]))

        print(f"🔄 SELL {opposite.upper()} {opp_tokens:.2f}tok @ {sell_prob:.1%}"
              f" → ${sell_revenue:.2f} | PnL {sell_order['pnl']:+.2f}"
              f" | Bal=${live['balance']:.2f}")

        _trades_writer.writerow([
            sell_order["time_str"], live["current_slug"],
            f"sell_{opposite}", f"{sell_revenue:.2f}",
            f"{sell_prob:.4f}", f"{btc:.2f}" if btc else "",
            sell_order["reason"], "sold", f"{sell_revenue:.2f}",
            f"{sell_order['pnl']:.2f}",
        ])
        _trades_file.flush()
        log_balance(f"sell_{opposite}")

        # Per-market CSV: SELL event row.
        write_market_row(
            event_kind="sell", event_side=opposite,
            bet_usd=0.0, buy_prob=sell_prob, tokens=opp_tokens,
            btc_at_event=btc if btc is not None else "",
            reason=sell_order["reason"], outcome="sold",
            payout=sell_revenue, pnl=sell_order["pnl"],
        )

    # ── Step 2: Buy on the target side ────────────────────────────────
    with lock:
        if live["balance"] < bet_usd:
            return  # sold but still can't afford

    buy_prob = up_prob if side == "up" else down_prob
    tokens = bet_usd / buy_prob if buy_prob > 0 else 0

    order = {
        "time": now,
        "time_str": datetime.now().strftime("%H:%M:%S"),
        "action": "BUY",
        "side": side,
        "bet_usd": bet_usd,
        "buy_prob": buy_prob,
        "tokens": tokens,
        "btc_price": btc,
        "up_prob": up_prob,
        "reason": reason,
        "outcome": None,
        "payout": None,
        "pnl": None,
        "slug": live["current_slug"],
    }

    tokens_key = f"{side}_tokens"
    cost_key = f"{side}_cost"

    with lock:
        live["balance"] -= bet_usd
        live[tokens_key] += tokens
        live[cost_key] += bet_usd
        live["orders"].append(order)
        live["all_orders"].append(order)
        live["last_order_time"] = now
        live["total_bets"] += 1
        live["balance_history"].append((now, live["balance"]))

    emoji = "🟢" if side == "up" else "🔴"
    print(f"{emoji} BUY {side.upper()} ${bet_usd:.2f} @ {buy_prob:.1%} → {tokens:.2f}tok"
          f" | BTC=${btc:,.2f} | {reason} | Bal=${live['balance']:.2f}"
          f" | Position: {live[tokens_key]:.2f} {side} tokens")

    _trades_writer.writerow([
        order["time_str"], live["current_slug"], f"buy_{side}", f"{bet_usd:.2f}",
        f"{buy_prob:.4f}", f"{btc:.2f}" if btc else "", reason, "", "", "",
    ])
    _trades_file.flush()
    log_balance(f"buy_{side}")

    # Per-market CSV: BUY event row.
    write_market_row(
        event_kind="buy", event_side=side,
        bet_usd=bet_usd, buy_prob=buy_prob, tokens=tokens,
        btc_at_event=btc if btc is not None else "",
        reason=reason, outcome="", payout="", pnl="",
    )


def resolve_market(btc_went_up: bool):
    return

    """
    Market ended. Winning tokens pay $1 each, losing tokens pay $0.
    Uses position tracker (up_tokens/down_tokens) not individual orders.
    """
    with lock:
        up_tok = live["up_tokens"]
        down_tok = live["down_tokens"]
        up_cost = live["up_cost"]
        down_cost = live["down_cost"]
        slug = live["current_slug"]
        orders = live["orders"]

    if up_tok == 0 and down_tok == 0 and not orders:
        with lock:
            live["last_resolution"] = f"{slug}: no positions"
        return

    # Winning side's tokens each pay $1
    up_payout = up_tok * 1.0 if btc_went_up else 0.0
    down_payout = down_tok * 1.0 if not btc_went_up else 0.0
    total_payout = up_payout + down_payout
    total_cost = up_cost + down_cost
    net_pnl = total_payout - total_cost

    # Mark individual BUY orders with outcome for the order log display,
    # and write a paired resolution row to trades.csv per BUY so the file
    # contains the outcome (the original BUY entry rows have empty outcome
    # fields because they were flushed at entry time).
    resolve_ts = datetime.now().strftime("%H:%M:%S")
    for o in orders:
        if o.get("action") == "BUY":
            won = (o["side"] == "up" and btc_went_up) or (o["side"] == "down" and not btc_went_up)
            o["outcome"] = "win" if won else "loss"
            if won:
                o["payout"] = o["tokens"] * 1.0
                o["pnl"] = o["payout"] - o["bet_usd"]
            else:
                o["payout"] = 0.0
                o["pnl"] = -o["bet_usd"]

            # Write resolution row: same shape as the BUY entry row but with
            # side prefixed "resolve_" and the outcome/payout/pnl filled in.
            _trades_writer.writerow([
                resolve_ts, slug, f"resolve_{o['side']}",
                f"{o['bet_usd']:.2f}", f"{o['buy_prob']:.4f}",
                f"{o['btc_price']:.2f}" if o.get("btc_price") else "",
                o["reason"], o["outcome"],
                f"{o['payout']:.2f}", f"{o['pnl']:.2f}",
            ])
    _trades_file.flush()

    with lock:
        live["balance"] += total_payout
        live["total_pnl"] += net_pnl
        live["total_wins"] += (1 if net_pnl > 0 else 0)
        live["balance_history"].append((time.time(), live["balance"]))

        # Reset positions
        live["up_tokens"] = 0.0
        live["down_tokens"] = 0.0
        live["up_cost"] = 0.0
        live["down_cost"] = 0.0

        winner = "UP" if btc_went_up else "DOWN"
        live["last_resolution"] = (
            f"{winner} wins | "
            f"Up: {up_tok:.1f}tok→${up_payout:.2f}  "
            f"Down: {down_tok:.1f}tok→${down_payout:.2f} | "
            f"PnL: {net_pnl:+.2f}"
        )

    print(f"RESOLVED {slug}: {winner} ✅ | "
          f"Up {up_tok:.1f}tok(${up_cost:.2f})→${up_payout:.2f}  "
          f"Down {down_tok:.1f}tok(${down_cost:.2f})→${down_payout:.2f} | "
          f"PnL {net_pnl:+.2f} | Bal ${live['balance']:.2f}")
    log_balance(f"resolve_{winner.lower()}")

    # Per-market CSV: final resolve row. The reason carries pre-resolve
    # token amounts so the row is self-describing for replay/training.
    write_market_row(
        event_kind="resolve", event_side=winner.lower(),
        bet_usd=total_cost, buy_prob="", tokens=up_tok + down_tok,
        btc_at_event="",
        reason=f"up_tok={up_tok:.4f} down_tok={down_tok:.4f}",
        outcome=winner.lower(),
        payout=total_payout, pnl=net_pnl,
    )


# ══════════════════════════════════════════════════════════════════════════════
# RULE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

def evaluate_rules():
    return None, "recorder only"

    r = RULES
    now = time.time()

    with lock:
        btc_series = list(live["btc_chainlink"]) or list(live["btc_binance"])
        up_series = list(live["up_price"])
        ptb = live["price_to_beat"]
        market_start = live["market_start"]
        market_end = live["market_end"]
        last_order = live["last_order_time"]
        balance = live["balance"]

    if len(btc_series) < 2 or not up_series or ptb is None:
        return None, "insufficient data"

    bet_usd = r.base_bet_usd * r.bet_multiplier
    if balance < bet_usd:
        return None, "insufficient balance"

    price = btc_series[-1][1]
    up = up_series[-1][1]
    elapsed = now - market_start
    remaining = market_end - now

    if elapsed < r.min_elapsed_s:
        return None, f"too early ({elapsed:.0f}s)"
    if remaining < r.min_remaining_s:
        return None, f"too late ({remaining:.0f}s left)"
    if now - last_order < r.cooldown_s:
        return None, "cooldown"
    if not (r.min_up_prob <= up <= r.max_up_prob):
        return None, f"odds out of range ({up:.1%})"

    cutoff = now - r.momentum_window_s
    window = [(t, p) for t, p in btc_series if t >= cutoff]
    if len(window) < 2:
        return None, "thin window"

    pct = ((price - window[0][1]) / window[0][1]) * 100
    dist = abs(price - ptb) / ptb * 100
    above = price > ptb

    reasons = []
    if pct >= r.momentum_pct:
        reasons.append(f"mom +{pct:.3f}%")
    if dist <= r.target_proximity_pct and above:
        reasons.append(f"near +{dist:.4f}%")
    if reasons:
        return "up", " & ".join(reasons)

    reasons = []
    if pct <= -r.momentum_pct:
        reasons.append(f"mom {pct:.3f}%")
    if dist <= r.target_proximity_pct and not above:
        reasons.append(f"near -{dist:.4f}%")
    if reasons:
        return "down", " & ".join(reasons)

    return None, f"no signal (Δ={pct:+.3f}%)"


# ══════════════════════════════════════════════════════════════════════════════
# DATA FEEDS
# ══════════════════════════════════════════════════════════════════════════════

def get_market(slot=None):
    now = int(time.time())
    if slot is None:
        slot = now - (now % 300)
    for offset in [0, 300, -300, 600]:
        s = slot + offset
        slug = f"btc-updown-5m-{s}"
        try:
            resp = requests.get(f"{GAMMA_API}/events?slug={slug}", timeout=5)
            data = resp.json()
        except:
            continue
        if data:
            ev = data[0]
            m = ev["markets"][0]
            tids = json.loads(m["clobTokenIds"]) if isinstance(m["clobTokenIds"], str) else m["clobTokenIds"]
            end = datetime.fromisoformat(m["endDate"].replace("Z", "+00:00")).timestamp()
            start = datetime.fromisoformat(ev["startTime"].replace("Z", "+00:00")).timestamp()
            if end > now:
                return {"slug": slug, "slot": s, "question": m["question"],
                        "up_token": tids[0], "down_token": tids[1],
                        "start_time": start, "end_time": end}
    return None


def poll_clob_loop():
    global _tick_file, _tick_writer
    current_market = None

    while live["running"]:
        now = time.time()

        if current_market is None or now >= current_market["end_time"]:
            if current_market is not None:
                with _tick_lock:
                    if _tick_file:
                        _tick_file.close()
                        _tick_file = None
                        _tick_writer = None

            ns = (current_market["slot"] + 300) if current_market else None
            retries = 0
            while retries < 12:
                current_market = get_market(ns)
                if current_market:
                    break
                time.sleep(5)
                retries += 1
                ns = None
            if not current_market:
                print("No market found. Stopping.")
                live["running"] = False
                return

            tick_dir = daily_data_dir(current_market["start_time"])
            tick_path = os.path.join(tick_dir, f"{current_market['slug']}.csv")
            with _tick_lock:
                _tick_file = open(tick_path, "w", newline="", encoding="utf-8")
                _tick_writer = csv.writer(_tick_file)
                _tick_writer.writerow(MARKET_CSV_HEADER)

            with lock:
                live["current_slug"] = current_market["slug"]
                live["market_question"] = current_market["question"]
                live["price_to_beat"] = None
                live["market_start"] = current_market["start_time"]
                live["market_end"] = current_market["end_time"]
                live["up_price"].clear()
                live["down_price"].clear()
                live["btc_binance"].clear()
                live["btc_chainlink"].clear()
                live["markets_seen"] += 1

            print(f"\n{'='*60}")
            print(f"Market #{live['markets_seen']}: {current_market['question']}")
            print(f"Recording: {tick_path}")
            print(f"{'='*60}")

            wait = current_market["start_time"] - time.time()
            if wait > 0:
                time.sleep(max(0, wait))

        remaining = current_market["end_time"] - time.time()
        if remaining <= 0:
            continue

        try:
            r = requests.get(f"{CLOB_API}/price",
                             params={"token_id": current_market["up_token"], "side": "buy"}, timeout=3)
            up_price = float(r.json().get("price", 0))
            down_price = round(1 - up_price, 3)
        except:
            time.sleep(0.05)
            continue

        now_t = time.time()
        with lock:
            live["up_price"].append((now_t, up_price))
            live["down_price"].append((now_t, down_price))
            if live["price_to_beat"] is None and live["btc_chainlink"] and now_t >= live["market_start"]:
                live["price_to_beat"] = live["btc_chainlink"][-1][1]

        # Tick row: state-only, no event columns.
        write_market_row()

        # Adaptive polling — faster as market end approaches
        if remaining <= 20:
            sleep_s = 0.05
        elif remaining <= 60:
            sleep_s = 0.08
        else:
            sleep_s = 0.10
        time.sleep(sleep_s)

    with _tick_lock:
        if _tick_file:
            _tick_file.close()
            _tick_file = None
            _tick_writer = None


def run_binance():
    async def connect():
        while live["running"]:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(BINANCE_WS) as ws:
                        print("✅ Binance connected")
                        async for msg in ws:
                            if not live["running"]:
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                d = json.loads(msg.data)
                                with lock:
                                    live["btc_binance"].append((time.time(), float(d["c"])))
            except Exception as e:
                print(f"Binance error: {e}")
                await asyncio.sleep(5)
    asyncio.run(connect())


def run_rtds():
    async def connect():
        while live["running"]:
            try:
                async with websockets.connect(RTDS_URL) as ws:
                    await ws.send(json.dumps({
                        "action": "subscribe",
                        "subscriptions": [{"topic": "crypto_prices_chainlink", "type": "*", "filters": ""}]
                    }))
                    print("✅ Chainlink/RTDS connected")
                    async def ping():
                        while live["running"]:
                            try:
                                await ws.send("PING")
                                await asyncio.sleep(5)
                            except:
                                break
                    ping_task = asyncio.create_task(ping())
                    async for msg in ws:
                        if not live["running"]:
                            break
                        if msg == "PONG" or not msg.strip():
                            continue
                        try:
                            d = json.loads(msg)
                        except:
                            continue
                        p = d.get("payload", {})
                        if p.get("symbol") == "btc/usd" and p.get("value") is not None:
                            with lock:
                                live["btc_chainlink"].append((time.time(), float(p["value"])))
                    ping_task.cancel()
            except Exception as e:
                print(f"RTDS error: {e}")
                await asyncio.sleep(5)
    asyncio.run(connect())


# ══════════════════════════════════════════════════════════════════════════════
# FLASK + SOCKETIO DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════

app = Flask(__name__)
app.config["SECRET_KEY"] = "dev"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")

DASHBOARD_HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>BTC Trader Dashboard</title>
<script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.1/dist/chart.umd.min.js"></script>
<script src="https://cdn.jsdelivr.net/npm/chartjs-plugin-annotation@3.0.1/dist/chartjs-plugin-annotation.min.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body {
    font-family: 'SF Mono', 'Fira Code', 'Consolas', monospace;
    background: #0a0a0f;
    color: #e0e0e0;
    overflow-x: hidden;
  }

  /* ── Top bar ─────────────────────────────────────────── */
  .topbar {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 12px 20px;
    background: #111118;
    border-bottom: 1px solid #1e1e2e;
  }
  .topbar .title {
    font-size: 14px;
    font-weight: 700;
    color: #8b8bff;
  }
  .topbar .question {
    font-size: 13px;
    color: #aaa;
    max-width: 500px;
    text-align: center;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  /* ── Countdown ───────────────────────────────────────── */
  .countdown-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
  }
  .countdown {
    font-size: 28px;
    font-weight: 800;
    font-variant-numeric: tabular-nums;
    min-width: 70px;
    text-align: right;
  }
  .countdown.warn  { color: #ff9800; }
  .countdown.crit  { color: #f44336; animation: pulse 0.5s infinite alternate; }
  .countdown.ok    { color: #4caf50; }
  .countdown-label { font-size: 10px; color: #666; text-transform: uppercase; }

  @keyframes pulse { to { opacity: 0.5; } }

  /* ── Stats row ───────────────────────────────────────── */
  .stats {
    display: flex;
    gap: 0;
    padding: 0 20px;
    background: #0d0d14;
    border-bottom: 1px solid #1a1a2a;
  }
  .stat {
    flex: 1;
    padding: 10px 16px;
    border-right: 1px solid #1a1a2a;
    text-align: center;
  }
  .stat:last-child { border-right: none; }
  .stat .label { font-size: 9px; color: #555; text-transform: uppercase; letter-spacing: 1px; }
  .stat .value { font-size: 20px; font-weight: 700; margin-top: 2px; font-variant-numeric: tabular-nums; }
  .stat .value.green { color: #4caf50; }
  .stat .value.red   { color: #f44336; }
  .stat .value.blue  { color: #42a5f5; }
  .stat .value.orange { color: #ff9800; }

  /* ── Charts ──────────────────────────────────────────── */
  .charts {
    display: grid;
    grid-template-columns: 1fr;
    gap: 0;
  }
  .chart-box {
    padding: 12px 16px 8px;
    border-bottom: 1px solid #1a1a2a;
    position: relative;
  }
  .chart-box canvas { width: 100% !important; }
  .chart-label {
    position: absolute;
    top: 14px;
    left: 20px;
    font-size: 10px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 1px;
    z-index: 2;
  }

  /* ── Order log ───────────────────────────────────────── */
  .orders-section {
    padding: 12px 20px;
  }
  .orders-section h3 {
    font-size: 11px;
    color: #555;
    text-transform: uppercase;
    letter-spacing: 1px;
    margin-bottom: 8px;
  }
  .order-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 12px;
  }
  .order-table th {
    text-align: left;
    padding: 6px 8px;
    color: #444;
    font-weight: 600;
    border-bottom: 1px solid #1a1a2a;
    font-size: 10px;
    text-transform: uppercase;
  }
  .order-table td {
    padding: 5px 8px;
    border-bottom: 1px solid #111118;
    font-variant-numeric: tabular-nums;
  }
  .order-table tr.up td:first-child { border-left: 3px solid #4caf50; }
  .order-table tr.down td:first-child { border-left: 3px solid #f44336; }
  .order-table .win { color: #4caf50; }
  .order-table .loss { color: #f44336; }
  .order-table .pending { color: #666; }

  /* ── Rules bar ───────────────────────────────────────── */
  .rules-bar {
    padding: 8px 20px;
    background: #0d0d14;
    border-top: 1px solid #1a1a2a;
    font-size: 11px;
    color: #444;
    display: flex;
    gap: 20px;
    flex-wrap: wrap;
  }
  .rules-bar span { color: #555; }

  .resolution-banner {
    padding: 6px 20px;
    background: #111118;
    border-bottom: 1px solid #1a1a2a;
    font-size: 12px;
    color: #888;
    text-align: center;
  }
</style>
</head>
<body>

<div class="topbar">
  <div class="title">BTC MOCK TRADER</div>
  <div class="question" id="question">Connecting...</div>
  <div class="countdown-wrap">
    <div>
      <div class="countdown ok" id="countdown">--</div>
      <div class="countdown-label">seconds left</div>
    </div>
  </div>
</div>

<div class="stats">
  <div class="stat">
    <div class="label">Balance</div>
    <div class="value orange" id="balance">$--</div>
  </div>
  <div class="stat">
    <div class="label">Total P&L</div>
    <div class="value" id="pnl">--</div>
  </div>
  <div class="stat">
    <div class="label">Win Rate</div>
    <div class="value blue" id="winrate">--</div>
  </div>
  <div class="stat">
    <div class="label">Markets</div>
    <div class="value" id="markets" style="color:#888">0</div>
  </div>
  <div class="stat">
    <div class="label">BTC (Chainlink)</div>
    <div class="value blue" id="btc_cl">--</div>
  </div>
  <div class="stat">
    <div class="label">Target</div>
    <div class="value" id="target" style="color:#f44336">--</div>
  </div>
  <div class="stat">
    <div class="label">Up / Down</div>
    <div class="value" id="updown">--</div>
  </div>
  <div class="stat">
    <div class="label">Bet Size</div>
    <div class="value" id="betsize" style="color:#888">--</div>
  </div>
  <div class="stat">
    <div class="label">Position</div>
    <div class="value" id="position" style="font-size:14px">--</div>
  </div>
</div>

<div class="resolution-banner" id="resolution">Waiting for first market...</div>

<div class="charts">
  <div class="chart-box" style="height:260px">
    <div class="chart-label">Bitcoin Price</div>
    <canvas id="btcChart"></canvas>
  </div>
  <div class="chart-box" style="height:200px">
    <div class="chart-label">Up / Down Probability</div>
    <canvas id="oddsChart"></canvas>
  </div>
</div>

<div class="orders-section">
  <h3>Order Log (recent 20)</h3>
  <table class="order-table">
    <thead>
      <tr>
        <th>Time</th>
        <th>Side</th>
        <th>Bet</th>
        <th>@ Prob</th>
        <th>BTC</th>
        <th>Reason</th>
        <th>Result</th>
        <th>PnL</th>
      </tr>
    </thead>
    <tbody id="orderBody"></tbody>
  </table>
</div>

<div class="rules-bar" id="rulesBar"></div>

<script>
const socket = io();

// ── Chart setup ──────────────────────────────────────────────
const chartOpts = (yLabel, suggestedMin, suggestedMax) => ({
  responsive: true,
  maintainAspectRatio: false,
  animation: false,
  interaction: { intersect: false, mode: 'index' },
  plugins: {
    legend: { display: true, position: 'top', labels: { color: '#666', font: { size: 10 }, boxWidth: 12 } },
    annotation: { annotations: {} },
  },
  scales: {
    x: { type: 'linear', display: true, ticks: { color: '#333', maxTicksLimit: 10,
         callback: v => v.toFixed(0) + 's' }, grid: { color: '#151520' },
         title: { display: false } },
    y: { ticks: { color: '#555' }, grid: { color: '#151520' },
         title: { display: true, text: yLabel, color: '#444', font: { size: 10 } },
         suggestedMin, suggestedMax },
  },
});

const btcChart = new Chart(document.getElementById('btcChart'), {
  type: 'line',
  data: {
    datasets: [
      { label: 'Chainlink', borderColor: '#42a5f5', borderWidth: 1.5, pointRadius: 0, data: [] },
      { label: 'Binance', borderColor: '#ff9800', borderWidth: 1, pointRadius: 0, data: [], borderDash: [2,2] },
    ]
  },
  options: chartOpts('BTC Price ($)', undefined, undefined),
});

const oddsChart = new Chart(document.getElementById('oddsChart'), {
  type: 'line',
  data: {
    datasets: [
      { label: 'Up %', borderColor: '#4caf50', borderWidth: 2, pointRadius: 0, data: [], fill: false },
      { label: 'Down %', borderColor: '#f44336', borderWidth: 2, pointRadius: 0, data: [], fill: false },
    ]
  },
  options: chartOpts('Probability (%)', 0, 100),
});

// ── Socket data handler ──────────────────────────────────────
socket.on('tick', d => {
  // Stats
  document.getElementById('question').textContent = d.question || 'Waiting...';
  document.getElementById('balance').textContent = '$' + d.balance.toFixed(2);

  const pnlEl = document.getElementById('pnl');
  pnlEl.textContent = (d.total_pnl >= 0 ? '+' : '') + d.total_pnl.toFixed(2);
  pnlEl.className = 'value ' + (d.total_pnl >= 0 ? 'green' : 'red');

  const wr = d.total_bets > 0 ? d.total_wins + '/' + d.total_bets : '—';
  document.getElementById('winrate').textContent = wr;
  document.getElementById('markets').textContent = d.markets_seen;

  const btcCl = d.btc_chainlink_latest;
  document.getElementById('btc_cl').textContent = btcCl ? '$' + btcCl.toLocaleString(undefined, {maximumFractionDigits: 0}) : '--';

  const ptb = d.price_to_beat;
  document.getElementById('target').textContent = ptb ? '$' + ptb.toLocaleString(undefined, {maximumFractionDigits: 0}) : '--';

  const upP = d.up_latest, dnP = d.down_latest;
  if (upP !== null) {
    document.getElementById('updown').innerHTML =
      '<span style="color:#4caf50">' + (upP * 100).toFixed(1) + '%</span>' +
      ' <span style="color:#444">/</span> ' +
      '<span style="color:#f44336">' + (dnP * 100).toFixed(1) + '%</span>';
  }

  document.getElementById('betsize').textContent = '$' + d.bet_size.toFixed(2);

  // Position
  const posEl = document.getElementById('position');
  if (d.up_tokens > 0) {
    posEl.innerHTML = '<span style="color:#4caf50">' + d.up_tokens.toFixed(1) + ' UP</span>';
  } else if (d.down_tokens > 0) {
    posEl.innerHTML = '<span style="color:#f44336">' + d.down_tokens.toFixed(1) + ' DN</span>';
  } else {
    posEl.innerHTML = '<span style="color:#444">flat</span>';
  }

  // Countdown
  const cdEl = document.getElementById('countdown');
  const rem = d.remaining;
  if (rem !== null && rem > 0) {
    cdEl.textContent = rem.toFixed(0);
    cdEl.className = 'countdown ' + (rem > 60 ? 'ok' : rem > 15 ? 'warn' : 'crit');
  } else {
    cdEl.textContent = '--';
    cdEl.className = 'countdown ok';
  }

  // Resolution
  if (d.last_resolution) {
    document.getElementById('resolution').textContent = 'Last: ' + d.last_resolution;
  }

  // Charts
  const ref = d.ref_time;
  if (ref) {
    btcChart.data.datasets[0].data = d.btc_chainlink.map(p => ({x: p[0] - ref, y: p[1]}));
    btcChart.data.datasets[1].data = d.btc_binance.map(p => ({x: p[0] - ref, y: p[1]}));

    // Target line
    const annots = {};
    if (ptb) {
      annots.target = { type: 'line', yMin: ptb, yMax: ptb, borderColor: '#f44336',
                         borderWidth: 1.5, borderDash: [6,3],
                         label: { display: true, content: 'Target $' + ptb.toFixed(0),
                                  color: '#f44336', font: {size: 9}, position: 'start' } };
    }
    // Order markers
    d.orders.forEach((o, i) => {
      if (o.btc_price) {
        const isSell = o.action === 'SELL';
        annots['ord'+i] = {
          type: 'point', xValue: o.time - ref, yValue: o.btc_price,
          backgroundColor: isSell ? '#ff9800' : (o.side === 'up' ? '#4caf50' : '#f44336'),
          borderColor: isSell ? '#fff' : '#fff',
          borderWidth: 1.5,
          radius: isSell ? 5 : 6,
          pointStyle: isSell ? 'rectRot' : 'circle',
        };
      }
    });
    btcChart.options.plugins.annotation.annotations = annots;
    btcChart.update('none');

    oddsChart.data.datasets[0].data = d.up_price.map(p => ({x: p[0] - ref, y: p[1] * 100}));
    oddsChart.data.datasets[1].data = d.down_price.map(p => ({x: p[0] - ref, y: p[1] * 100}));

    const oddsAnnots = {};
    oddsAnnots.fifty = { type: 'line', yMin: 50, yMax: 50, borderColor: '#333',
                          borderWidth: 1, borderDash: [4,4] };
    d.orders.forEach((o, i) => {
      oddsAnnots['vline'+i] = {
        type: 'line', xMin: o.time - ref, xMax: o.time - ref,
        borderColor: o.side === 'up' ? '#4caf5088' : '#f4433688', borderWidth: 1.5,
      };
    });
    oddsChart.options.plugins.annotation.annotations = oddsAnnots;
    oddsChart.update('none');
  }

  // Order table
  const tbody = document.getElementById('orderBody');
  const rows = d.all_orders.slice(-20).reverse();
  tbody.innerHTML = rows.map(o => {
    const action = o.action || 'BUY';
    const cls = o.side;
    const isSell = action === 'SELL';
    const actionLabel = isSell ? '🔄 SELL' : (o.side === 'up' ? '🟢 BUY' : '🔴 BUY');
    const res = isSell ? '<span style="color:#888">sold</span>' :
                o.outcome === 'win' ? '<span class="win">WIN</span>' :
                o.outcome === 'loss' ? '<span class="loss">LOSS</span>' :
                '<span class="pending">pending</span>';
    const pnl = o.pnl !== null ?
      (o.pnl >= 0 ? '<span class="win">+' + o.pnl.toFixed(2) + '</span>' :
                     '<span class="loss">' + o.pnl.toFixed(2) + '</span>') : '—';
    return '<tr class="' + cls + '">' +
      '<td>' + o.time_str + '</td>' +
      '<td>' + actionLabel + ' ' + o.side.toUpperCase() + '</td>' +
      '<td>$' + (o.bet_usd || 0).toFixed(2) + '</td>' +
      '<td>' + (o.buy_prob * 100).toFixed(1) + '%</td>' +
      '<td>$' + (o.btc_price ? o.btc_price.toLocaleString(undefined,{maximumFractionDigits:0}) : '--') + '</td>' +
      '<td>' + o.reason + '</td>' +
      '<td>' + res + '</td>' +
      '<td>' + pnl + '</td></tr>';
  }).join('');
});

socket.on('connect', () => console.log('Connected'));
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


def emit_tick():
    """Snapshot live state and push to all connected browsers."""
    with lock:
        # Downsample for the browser — keep last 600 points max per series.
        # Slice copies are taken under the lock so concurrent appends from
        # the data feeds and execute_order can't mutate them mid-serialise.
        MAX = 600
        btc_b = live["btc_binance"][-MAX:]
        btc_c = live["btc_chainlink"][-MAX:]
        up_p  = live["up_price"][-MAX:]
        dn_p  = live["down_price"][-MAX:]

        # Orders are dicts that get mutated in-place at resolution time
        # (outcome/payout/pnl), so we shallow-copy each dict — not just the
        # list. Without this, a SELL → BUY pair from execute_order could
        # mutate the payload while socketio is JSON-serialising it, which
        # presents to the browser as the chart freezing on flips.
        orders_snapshot = []
        all_orders_snapshot = []

        ref = (up_p or btc_c or btc_b or [(0,0)])[0][0]
        remaining = live["market_end"] - time.time() if live["market_end"] else None

        payload = {
            "question": live["market_question"],
            "slug": live["current_slug"],
            "balance": 0.0,
            "total_pnl": 0.0,
            "total_bets": 0,
            "total_wins": 0,
            "markets_seen": live["markets_seen"],
            "price_to_beat": live["price_to_beat"],
            "remaining": remaining,
            "last_resolution": "recorder only",
            "bet_size": 0.0,
            "ref_time": ref,

            "btc_binance_latest": btc_b[-1][1] if btc_b else None,
            "btc_chainlink_latest": btc_c[-1][1] if btc_c else None,
            "up_latest": up_p[-1][1] if up_p else None,
            "down_latest": dn_p[-1][1] if dn_p else None,

            "up_tokens": 0.0,
            "down_tokens": 0.0,

            "btc_binance": btc_b,
            "btc_chainlink": btc_c,
            "up_price": up_p,
            "down_price": dn_p,
            "orders": orders_snapshot,
            "all_orders": all_orders_snapshot,
        }

    socketio.emit("tick", payload)


def tick_loop():
    """Push updates to the browser every 500ms."""
    while live["running"]:
        try:
            emit_tick()
        except Exception as e:
            pass
        time.sleep(0.5)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    bet = RULES.base_bet_usd * RULES.bet_multiplier

    print("BTC Market Recorder")
    print("=" * 50)
    print(f"Bet size:         ${bet:.2f}  (${RULES.base_bet_usd} × {RULES.bet_multiplier})")
    print(f"Momentum:         ≥{RULES.momentum_pct}% in {RULES.momentum_window_s}s")
    print(f"Proximity:        ≤{RULES.target_proximity_pct}% of target")
    print(f"Odds range:       {RULES.min_up_prob:.0%} – {RULES.max_up_prob:.0%}")
    print(f"Cooldown:         {RULES.cooldown_s}s")
    print(f"{'='*50}")
    print(f"Dashboard:        http://localhost:5050")
    print(f"Saves:            {DATA_DIR}/YYYY-MM-DD/<slug>.csv")
    print()

    threading.Thread(target=poll_clob_loop, daemon=True).start()
    threading.Thread(target=run_binance, daemon=True).start()
    threading.Thread(target=run_rtds, daemon=True).start()
    threading.Thread(target=tick_loop, daemon=True).start()

    # Flask runs on main thread
    socketio.run(app, host="0.0.0.0", port=5050, debug=False, allow_unsafe_werkzeug=True)
