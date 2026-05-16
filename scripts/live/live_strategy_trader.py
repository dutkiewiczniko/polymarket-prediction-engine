import argparse
import asyncio
import csv
import json
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import aiohttp
import requests
import websockets
from flask import Flask, render_template_string
from flask_socketio import SocketIO

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.batch import resolve_effective_market_balance
from simulator.config_loader import build_strategy_from_config, load_yaml
from simulator.execution import execute_action
from simulator.models import DecisionState, MarketTick
from simulator.portfolio import Portfolio


GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
RTDS_URL = "wss://ws-live-data.polymarket.com"
BINANCE_WS = "wss://stream.binance.com:9443/ws/btcusdt@ticker"

MARKET_DATA_DIR = Path("data")
LIVE_RUN_DIR = Path("runs/live_strategy")
ACCOUNT_HISTORY_PATH = LIVE_RUN_DIR / "account_history.csv"
MARKET_CSV_HEADER = [
    "timestamp",
    "unix_time",
    "seconds_left",
    "elapsed",
    "up_price",
    "down_price",
    "btc_binance",
    "btc_chainlink",
    "price_to_beat",
]
TRAJECTORY_COLUMNS = [
    "timestamp",
    "unix_time",
    "elapsed",
    "seconds_left",
    "up_price",
    "down_price",
    "btc_price",
    "price_to_beat",
    "master_balance_before_market",
    "market_simulated_starting_balance",
    "cash_before",
    "up_tokens_before",
    "down_tokens_before",
    "balance_before",
    "action",
    "reason",
    "usd_amount",
    "events_count",
    "market_spend_used_before",
    "market_spend_used_after",
    "cash_after",
    "up_tokens_after",
    "down_tokens_after",
    "balance_after",
    "master_balance_after_market",
    "market_simulated_final_balance",
    "final_outcome",
    "total_reward",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Live paper trader for one rule-based strategy.")
    parser.add_argument(
        "--strategy-config",
        default="configs/strategies/down_bias_chaser_capped_token_compound/down_bias_chaser_tp095_cheap0015_tok20.yaml",
        help="Strategy YAML to run live.",
    )
    parser.add_argument(
        "--batch-config",
        default="configs/simulation_down_bias_chaser_capped_token_compound.yaml",
        help="Batch config providing starting balance and market-balance scaling rules.",
    )
    parser.add_argument("--port", type=int, default=5051, help="Local dashboard port.")
    return parser.parse_args()


def format_csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return value


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
        except Exception:
            continue
        if data:
            ev = data[0]
            m = ev["markets"][0]
            tids = json.loads(m["clobTokenIds"]) if isinstance(m["clobTokenIds"], str) else m["clobTokenIds"]
            end = datetime.fromisoformat(m["endDate"].replace("Z", "+00:00")).timestamp()
            start = datetime.fromisoformat(ev["startTime"].replace("Z", "+00:00")).timestamp()
            if end > now:
                return {
                    "slug": slug,
                    "slot": s,
                    "question": m["question"],
                    "up_token": tids[0],
                    "down_token": tids[1],
                    "start_time": start,
                    "end_time": end,
                }
    return None


class LiveStrategyTrader:
    def __init__(self, args):
        self.args = args
        self.batch_cfg = load_yaml(args.batch_config)
        self.strategy_cfg = load_yaml(args.strategy_config)
        self.base_starting_balance = float(self.batch_cfg.get("starting_balance", self.strategy_cfg.get("starting_balance", 2000.0)))
        self.master_balance = self.base_starting_balance
        self.strategy_name = self.strategy_cfg.get("name", Path(args.strategy_config).stem)
        self.lock = threading.Lock()
        self.running = True

        MARKET_DATA_DIR.mkdir(parents=True, exist_ok=True)
        LIVE_RUN_DIR.mkdir(parents=True, exist_ok=True)

        self.state = {
            "btc_binance": [],
            "btc_chainlink": [],
            "up_price": [],
            "down_price": [],
            "price_to_beat": None,
            "market_question": "",
            "current_slug": "",
            "market_start": 0.0,
            "market_end": 0.0,
            "markets_seen": 0,
        }

        self.current_market = None
        self.strategy = None
        self.portfolio = None
        self.market_spend_used = 0.0
        self.master_balance_before_market = self.master_balance
        self.simulated_market_balance = self.base_starting_balance
        self.last_action = "none"
        self.orders_placed = 0
        self.trajectory_rows = []

        self.market_file = None
        self.market_writer = None
        self.recent_completed_markets = []

        self.ensure_account_history_header()

    def ensure_account_history_header(self):
        if ACCOUNT_HISTORY_PATH.exists():
            return
        with ACCOUNT_HISTORY_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "timestamp",
                "market_slug",
                "strategy_name",
                "master_balance_before_market",
                "market_simulated_starting_balance",
                "market_simulated_final_balance",
                "master_balance_after_market",
                "total_reward",
                "final_outcome",
            ])

    def append_account_history(self, final_balance, total_reward, final_outcome):
        with ACCOUNT_HISTORY_PATH.open("a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now().isoformat(timespec="seconds"),
                self.current_market["slug"] if self.current_market else "",
                self.strategy_name,
                format_csv_value(self.master_balance_before_market),
                format_csv_value(self.simulated_market_balance),
                format_csv_value(final_balance),
                format_csv_value(self.master_balance),
                format_csv_value(total_reward),
                final_outcome,
            ])

    def begin_market(self, market):
        self.end_market()
        self.current_market = market
        self.strategy = build_strategy_from_config(self.strategy_cfg)
        self.master_balance_before_market = self.master_balance
        effective_balance = resolve_effective_market_balance(self.master_balance_before_market, self.batch_cfg)
        self.simulated_market_balance = effective_balance if effective_balance is not None else self.master_balance_before_market
        self.portfolio = Portfolio(cash=self.simulated_market_balance)
        self.market_spend_used = 0.0
        self.last_action = "none"
        self.orders_placed = 0
        self.trajectory_rows = []

        market_path = MARKET_DATA_DIR / f"{market['slug']}.csv"
        self.market_file = market_path.open("w", newline="", encoding="utf-8")
        self.market_writer = csv.writer(self.market_file)
        self.market_writer.writerow(MARKET_CSV_HEADER)

        with self.lock:
            self.state["current_slug"] = market["slug"]
            self.state["market_question"] = market["question"]
            self.state["price_to_beat"] = None
            self.state["market_start"] = market["start_time"]
            self.state["market_end"] = market["end_time"]
            self.state["up_price"].clear()
            self.state["down_price"].clear()
            self.state["btc_binance"].clear()
            self.state["btc_chainlink"].clear()
            self.state["markets_seen"] += 1

        print()
        print("=" * 72)
        print(f"Market #{self.state['markets_seen']}: {market['question']}")
        print(f"Strategy: {self.strategy_name}")
        print(f"Master balance before market: {self.master_balance_before_market:.4f}")
        print(f"Simulated market starting balance: {self.simulated_market_balance:.4f}")
        print(f"Recording market: {market_path}")
        print("=" * 72)

    def end_market(self):
        if self.market_file:
            self.market_file.close()
            self.market_file = None
            self.market_writer = None

        if not self.current_market or not self.trajectory_rows or self.portfolio is None:
            self.current_market = None
            return

        final_outcome = self.infer_final_outcome()
        final_balance = self.portfolio.resolve(final_outcome)
        total_reward = final_balance - self.simulated_market_balance
        self.master_balance = max(0.0, self.master_balance_before_market + total_reward)

        for row in self.trajectory_rows:
            row["master_balance_after_market"] = self.master_balance
            row["market_simulated_final_balance"] = final_balance
            row["final_outcome"] = final_outcome
            row["total_reward"] = total_reward

        output_path = LIVE_RUN_DIR / f"{self.current_market['slug']}_{self.strategy_name}_trajectory.csv"
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRAJECTORY_COLUMNS)
            writer.writeheader()
            writer.writerows(self.trajectory_rows)

        self.append_account_history(final_balance, total_reward, final_outcome)
        self.recent_completed_markets.insert(0, {
            "market_slug": self.current_market["slug"],
            "final_outcome": final_outcome,
            "master_balance_before_market": self.master_balance_before_market,
            "market_simulated_starting_balance": self.simulated_market_balance,
            "market_simulated_final_balance": final_balance,
            "master_balance_after_market": self.master_balance,
            "total_reward": total_reward,
        })
        self.recent_completed_markets = self.recent_completed_markets[:12]

        print(
            f"{self.current_market['slug']}: outcome={final_outcome} "
            f"sim_final={final_balance:.4f} reward={total_reward:.4f} "
            f"master_after={self.master_balance:.4f}"
        )
        self.current_market = None

    def infer_final_outcome(self):
        last_btc = None
        last_ptb = None
        with self.lock:
            series = self.state["btc_chainlink"] or self.state["btc_binance"]
            for _, price in series:
                last_btc = price
            last_ptb = self.state["price_to_beat"]
        if last_btc is None or last_ptb is None:
            return "up"
        return "up" if last_btc >= last_ptb else "down"

    def latest_snapshot(self):
        now = time.time()
        with self.lock:
            market_start = self.state["market_start"]
            market_end = self.state["market_end"]
            up_price = self.state["up_price"][-1][1] if self.state["up_price"] else None
            down_price = self.state["down_price"][-1][1] if self.state["down_price"] else None
            btc_binance = self.state["btc_binance"][-1][1] if self.state["btc_binance"] else None
            btc_chainlink = self.state["btc_chainlink"][-1][1] if self.state["btc_chainlink"] else None
            price_to_beat = self.state["price_to_beat"]

        return {
            "timestamp": datetime.fromtimestamp(now).strftime("%H:%M:%S.%f")[:-3],
            "unix_time": now,
            "seconds_left": market_end - now if market_end else None,
            "elapsed": now - market_start if market_start else None,
            "up_price": up_price,
            "down_price": down_price,
            "btc_binance": btc_binance,
            "btc_chainlink": btc_chainlink,
            "price_to_beat": price_to_beat,
        }

    def append_market_row(self, row):
        if not self.market_writer or not self.market_file:
            return
        self.market_writer.writerow([format_csv_value(row.get(column)) for column in MARKET_CSV_HEADER])
        self.market_file.flush()

    def build_tick(self, row):
        return MarketTick(
            timestamp=row["timestamp"],
            unix_time=row["unix_time"],
            seconds_left=row["seconds_left"],
            elapsed=row["elapsed"],
            up_price=row["up_price"],
            down_price=row["down_price"],
            btc_binance=row["btc_binance"],
            btc_chainlink=row["btc_chainlink"],
            price_to_beat=row["price_to_beat"],
        )

    def step_strategy(self, row):
        if self.strategy is None or self.portfolio is None:
            return

        tick = self.build_tick(row)
        current_balance = self.portfolio.mark_to_market(row["up_price"], row["down_price"])
        state = DecisionState(
            tick=tick,
            cash=self.portfolio.cash,
            up_tokens=self.portfolio.up_tokens,
            down_tokens=self.portfolio.down_tokens,
            current_balance=current_balance,
            market_start_balance=self.master_balance_before_market,
            market_spend_used=self.market_spend_used,
            last_action=self.last_action,
            orders_placed=self.orders_placed,
        )

        decision = self.strategy.decide(state)
        usd_amount = decision.usd_amount if decision.usd_amount is not None else float(self.strategy_cfg.get("order_usd", 1.0))
        spend_before = self.market_spend_used
        events = execute_action(
            portfolio=self.portfolio,
            action=decision.action,
            timestamp=row["timestamp"],
            up_price=row["up_price"],
            down_price=row["down_price"],
            usd_amount=usd_amount,
            reason=decision.reason,
        )
        if events:
            self.orders_placed += len(events)
            self.market_spend_used += sum(event.usd_amount for event in events if event.action == "buy")

        balance_after = self.portfolio.mark_to_market(row["up_price"], row["down_price"])
        self.trajectory_rows.append({
            "timestamp": row["timestamp"],
            "unix_time": row["unix_time"],
            "elapsed": row["elapsed"],
            "seconds_left": row["seconds_left"],
            "up_price": row["up_price"],
            "down_price": row["down_price"],
            "btc_price": row["btc_chainlink"] if row["btc_chainlink"] is not None else row["btc_binance"],
            "price_to_beat": row["price_to_beat"],
            "master_balance_before_market": self.master_balance_before_market,
            "market_simulated_starting_balance": self.simulated_market_balance,
            "cash_before": state.cash,
            "up_tokens_before": state.up_tokens,
            "down_tokens_before": state.down_tokens,
            "balance_before": state.current_balance,
            "action": decision.action,
            "reason": decision.reason,
            "usd_amount": usd_amount,
            "events_count": len(events),
            "market_spend_used_before": spend_before,
            "market_spend_used_after": self.market_spend_used,
            "cash_after": self.portfolio.cash,
            "up_tokens_after": self.portfolio.up_tokens,
            "down_tokens_after": self.portfolio.down_tokens,
            "balance_after": balance_after,
            "master_balance_after_market": "",
            "market_simulated_final_balance": "",
            "final_outcome": "",
            "total_reward": "",
        })
        self.last_action = decision.action

        print(
            f"elapsed={row['elapsed']:.1f} "
            f"master={self.master_balance_before_market:.2f} sim={self.simulated_market_balance:.2f} "
            f"cash={self.portfolio.cash:.2f} action={decision.action} usd={usd_amount:.2f} "
            f"spend={self.market_spend_used:.2f} reason={decision.reason}"
        )

    def dashboard_payload(self):
        with self.lock:
            up_price = self.state["up_price"][-1][1] if self.state["up_price"] else None
            down_price = self.state["down_price"][-1][1] if self.state["down_price"] else None
            btc_price = self.state["btc_chainlink"][-1][1] if self.state["btc_chainlink"] else (
                self.state["btc_binance"][-1][1] if self.state["btc_binance"] else None
            )
            price_to_beat = self.state["price_to_beat"]
            market_question = self.state["market_question"]
            current_slug = self.state["current_slug"]
            market_end = self.state["market_end"]
            market_start = self.state["market_start"]
            markets_seen = self.state["markets_seen"]

        now = time.time()
        seconds_left = market_end - now if market_end else None
        elapsed = now - market_start if market_start else None
        distance = (btc_price - price_to_beat) if (btc_price is not None and price_to_beat is not None) else None
        distance_pct = (distance / price_to_beat) if (distance is not None and price_to_beat) else None
        current_balance = self.portfolio.mark_to_market(up_price or 0.0, down_price or 0.0) if self.portfolio else None
        latest = self.trajectory_rows[-1] if self.trajectory_rows else {}

        return {
            "strategy_name": self.strategy_name,
            "market_question": market_question,
            "current_slug": current_slug,
            "markets_seen": markets_seen,
            "seconds_left": seconds_left,
            "elapsed": elapsed,
            "up_price": up_price,
            "down_price": down_price,
            "btc_price": btc_price,
            "price_to_beat": price_to_beat,
            "distance": distance,
            "distance_pct": distance_pct,
            "master_balance": self.master_balance,
            "master_balance_before_market": self.master_balance_before_market,
            "market_simulated_starting_balance": self.simulated_market_balance,
            "current_market_balance": current_balance,
            "cash": self.portfolio.cash if self.portfolio else None,
            "up_tokens": self.portfolio.up_tokens if self.portfolio else None,
            "down_tokens": self.portfolio.down_tokens if self.portfolio else None,
            "orders_placed": self.orders_placed,
            "market_spend_used": self.market_spend_used,
            "latest_action": latest.get("action", ""),
            "latest_reason": latest.get("reason", ""),
            "latest_usd_amount": latest.get("usd_amount", ""),
            "recent_completed_markets": list(self.recent_completed_markets),
            "market_csv_path": str(MARKET_DATA_DIR / f"{current_slug}.csv") if current_slug else "",
        }


def poll_clob_loop(trader: LiveStrategyTrader):
    current_market = None
    while trader.running:
        now = time.time()
        if current_market is None or now >= current_market["end_time"]:
            if current_market is not None:
                trader.end_market()
            ns = (current_market["slot"] + 300) if current_market else None
            current_market = None
            for _ in range(12):
                current_market = get_market(ns)
                if current_market:
                    break
                ns = None
                time.sleep(5)
            if current_market is None:
                print("No market found yet.")
                time.sleep(5)
                continue
            trader.begin_market(current_market)
            wait = current_market["start_time"] - time.time()
            if wait > 0:
                time.sleep(wait)

        remaining = current_market["end_time"] - time.time()
        if remaining <= 0:
            continue

        try:
            r = requests.get(
                f"{CLOB_API}/price",
                params={"token_id": current_market["up_token"], "side": "buy"},
                timeout=3,
            )
            up_price = float(r.json().get("price", 0))
            down_price = round(1 - up_price, 3)
        except Exception as e:
            print(f"CLOB price error: {e}")
            time.sleep(0.1)
            continue

        now_t = time.time()
        with trader.lock:
            trader.state["up_price"].append((now_t, up_price))
            trader.state["down_price"].append((now_t, down_price))
            if (
                trader.state["price_to_beat"] is None
                and trader.state["btc_chainlink"]
            ):
                trader.state["price_to_beat"] = trader.state["btc_chainlink"][-1][1]

        row = trader.latest_snapshot()
        trader.append_market_row(row)
        if all(row.get(column) is not None for column in ["elapsed", "seconds_left", "up_price", "down_price", "price_to_beat"]) and (
            row.get("btc_chainlink") is not None or row.get("btc_binance") is not None
        ):
            trader.step_strategy(row)

        if remaining <= 20:
            sleep_s = 0.05
        elif remaining <= 60:
            sleep_s = 0.08
        else:
            sleep_s = 0.10
        time.sleep(sleep_s)


def start_binance_loop(trader: LiveStrategyTrader):
    async def connect():
        while trader.running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(BINANCE_WS) as ws:
                        print("Binance connected")
                        async for msg in ws:
                            if not trader.running:
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                price = float(data["c"])
                                with trader.lock:
                                    trader.state["btc_binance"].append((time.time(), price))
            except Exception as e:
                print(f"Binance reconnecting: {e}")
                await asyncio.sleep(2)

    asyncio.run(connect())


def start_chainlink_loop(trader: LiveStrategyTrader):
    async def connect():
        while trader.running:
            try:
                async with websockets.connect(RTDS_URL) as ws:
                    await ws.send(json.dumps({
                        "action": "subscribe",
                        "subscriptions": [{"topic": "crypto_prices_chainlink", "type": "*", "filters": ""}],
                    }))
                    print("Chainlink connected")
                    async for msg in ws:
                        if not trader.running:
                            break
                        if msg == "PONG" or not msg.strip():
                            continue
                        try:
                            payload = json.loads(msg)
                        except json.JSONDecodeError:
                            continue
                        if payload.get("asset") != "BTC":
                            continue
                        price = payload.get("price")
                        if price is None:
                            continue
                        with trader.lock:
                            trader.state["btc_chainlink"].append((time.time(), float(price)))
            except Exception as e:
                print(f"Chainlink reconnecting: {e}")
                await asyncio.sleep(2)

    asyncio.run(connect())


app = Flask(__name__)
app.config["SECRET_KEY"] = "dev"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
TRADER = None

DASHBOARD_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Live Strategy Trader</title>
  <script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
  <style>
    body { margin:0; font-family: Consolas, monospace; background:#0b0e13; color:#e8ecf3; }
    .top { display:flex; justify-content:space-between; align-items:flex-end; padding:14px 18px; background:#131923; border-bottom:1px solid #273246; }
    .title { font-size:18px; font-weight:700; color:#8ec1ff; }
    .sub { color:#9aa6b8; font-size:12px; }
    .grid { display:grid; grid-template-columns: repeat(4, 1fr); gap:1px; background:#273246; }
    .cell { background:#101722; padding:14px 16px; min-height:68px; }
    .label { color:#8d98aa; font-size:11px; text-transform:uppercase; }
    .value { font-size:20px; margin-top:6px; }
    .wide { grid-column: span 2; }
    .full { grid-column: 1 / -1; }
    .tables { display:grid; grid-template-columns: 1fr 1fr; gap:16px; padding:16px; }
    table { width:100%; border-collapse:collapse; background:#101722; }
    th, td { padding:8px 10px; border-bottom:1px solid #273246; font-size:12px; text-align:left; }
    th { color:#8ec1ff; }
    .green { color:#6fd38b; }
    .red { color:#ff7d7d; }
    .muted { color:#98a4b5; }
  </style>
</head>
<body>
  <div class="top">
    <div>
      <div class="title">Live Strategy Trader</div>
      <div class="sub" id="strategy"></div>
    </div>
    <div class="sub" id="marketcsv"></div>
  </div>
  <div class="grid">
    <div class="cell wide"><div class="label">Market</div><div class="value" id="market">--</div></div>
    <div class="cell"><div class="label">Markets Seen</div><div class="value" id="seen">--</div></div>
    <div class="cell"><div class="label">Time Left</div><div class="value" id="left">--</div></div>
    <div class="cell"><div class="label">Prices</div><div class="value" id="prices">--</div></div>
    <div class="cell"><div class="label">BTC</div><div class="value" id="btc">--</div></div>
    <div class="cell"><div class="label">Price To Beat</div><div class="value" id="ptb">--</div></div>
    <div class="cell"><div class="label">BTC Distance</div><div class="value" id="dist">--</div></div>
    <div class="cell"><div class="label">Master Balance</div><div class="value" id="master">--</div></div>
    <div class="cell"><div class="label">Master Before Market</div><div class="value" id="masterbefore">--</div></div>
    <div class="cell"><div class="label">Simulated Market Start</div><div class="value" id="simstart">--</div></div>
    <div class="cell"><div class="label">Current Market Balance</div><div class="value" id="simbal">--</div></div>
    <div class="cell"><div class="label">Cash / Tokens</div><div class="value" id="pos">--</div></div>
    <div class="cell"><div class="label">Orders Placed</div><div class="value" id="orders">--</div></div>
    <div class="cell"><div class="label">Market Spend Used</div><div class="value" id="spend">--</div></div>
    <div class="cell"><div class="label">Latest Action</div><div class="value" id="action">--</div></div>
    <div class="cell full"><div class="label">Latest Reason</div><div class="value" id="reason">--</div></div>
  </div>
  <div class="tables">
    <div>
      <h3>Previous Markets</h3>
      <table>
        <thead><tr><th>Market</th><th>Outcome</th><th>Reward</th><th>Master After</th></tr></thead>
        <tbody id="completed"></tbody>
      </table>
    </div>
    <div>
      <h3>Current / Last Market Stats</h3>
      <table>
        <thead><tr><th>Field</th><th>Value</th></tr></thead>
        <tbody>
          <tr><td>Elapsed</td><td id="elapsed">--</td></tr>
          <tr><td>Last USD Amount</td><td id="usd">--</td></tr>
          <tr><td>Current Slug</td><td id="slug">--</td></tr>
        </tbody>
      </table>
    </div>
  </div>
<script>
const socket = io();
function fmtMoney(x){ return x == null || x === '' ? '--' : Number(x).toFixed(2); }
function fmtNum(x, d=3){ return x == null || x === '' ? '--' : Number(x).toFixed(d); }
socket.on('tick', d => {
  document.getElementById('strategy').textContent = d.strategy_name || '--';
  document.getElementById('marketcsv').textContent = d.market_csv_path || '--';
  document.getElementById('market').textContent = d.market_question || '--';
  document.getElementById('seen').textContent = d.markets_seen ?? '--';
  document.getElementById('left').textContent = d.seconds_left == null ? '--' : Number(d.seconds_left).toFixed(1) + 's';
  document.getElementById('prices').innerHTML = '<span class="green">' + fmtNum(d.up_price) + '</span> / <span class="red">' + fmtNum(d.down_price) + '</span>';
  document.getElementById('btc').textContent = fmtMoney(d.btc_price);
  document.getElementById('ptb').textContent = fmtMoney(d.price_to_beat);
  document.getElementById('dist').textContent = d.distance == null ? '--' : fmtMoney(d.distance) + ' / ' + fmtNum((d.distance_pct || 0) * 100, 4) + '%';
  document.getElementById('master').textContent = fmtMoney(d.master_balance);
  document.getElementById('masterbefore').textContent = fmtMoney(d.master_balance_before_market);
  document.getElementById('simstart').textContent = fmtMoney(d.market_simulated_starting_balance);
  document.getElementById('simbal').textContent = fmtMoney(d.current_market_balance);
  document.getElementById('pos').textContent = 'cash ' + fmtMoney(d.cash) + ' / up ' + fmtNum(d.up_tokens,2) + ' / down ' + fmtNum(d.down_tokens,2);
  document.getElementById('orders').textContent = d.orders_placed ?? '--';
  document.getElementById('spend').textContent = fmtMoney(d.market_spend_used);
  document.getElementById('action').textContent = d.latest_action || '--';
  document.getElementById('reason').textContent = d.latest_reason || '--';
  document.getElementById('elapsed').textContent = d.elapsed == null ? '--' : Number(d.elapsed).toFixed(1) + 's';
  document.getElementById('usd').textContent = fmtMoney(d.latest_usd_amount);
  document.getElementById('slug').textContent = d.current_slug || '--';
  document.getElementById('completed').innerHTML = (d.recent_completed_markets || []).map(m =>
    '<tr><td>'+m.market_slug+'</td><td>'+m.final_outcome+'</td><td>'+fmtMoney(m.total_reward)+'</td><td>'+fmtMoney(m.master_balance_after_market)+'</td></tr>'
  ).join('');
});
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(DASHBOARD_HTML)


def emit_tick_loop():
    while TRADER and TRADER.running:
        socketio.emit("tick", TRADER.dashboard_payload())
        time.sleep(0.5)


def main():
    global TRADER
    args = parse_args()
    trader = LiveStrategyTrader(args)
    TRADER = trader

    print("Live strategy paper trader")
    print("=" * 50)
    print(f"Strategy config: {args.strategy_config}")
    print(f"Batch config:    {args.batch_config}")
    print(f"Start balance:   {trader.base_starting_balance}")
    print(f"Dashboard:       http://localhost:{args.port}")
    print(f"Data:            {MARKET_DATA_DIR}/<slug>.csv")
    print(f"Runs:            {LIVE_RUN_DIR}/<slug>_{trader.strategy_name}_trajectory.csv")
    print(f"Account log:     {ACCOUNT_HISTORY_PATH}")

    binance_thread = threading.Thread(target=start_binance_loop, args=(trader,), daemon=True)
    chainlink_thread = threading.Thread(target=start_chainlink_loop, args=(trader,), daemon=True)
    binance_thread.start()
    chainlink_thread.start()

    threading.Thread(target=poll_clob_loop, args=(trader,), daemon=True).start()
    threading.Thread(target=emit_tick_loop, daemon=True).start()

    try:
        socketio.run(app, host="0.0.0.0", port=args.port, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        trader.running = False
        trader.end_market()


if __name__ == "__main__":
    main()
