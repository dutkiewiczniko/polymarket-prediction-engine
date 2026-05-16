import argparse
import asyncio
import csv
import json
import math
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import aiohttp
import numpy as np
import pandas as pd
import requests
import websockets
from flask import Flask, render_template_string
from flask_socketio import SocketIO

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.execution import execute_action
from simulator.model_loader import find_latest_reward_model, load_reward_model
from simulator.portfolio import Portfolio
from scripts.simulation.ml_inference_replay import (
    build_market_feature_row,
    build_prediction_rows,
    portfolio_feature_row,
    validate_prediction_frame,
)
from scripts.training.train_reward_model import FEATURE_COLUMNS


GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
RTDS_URL = "wss://ws-live-data.polymarket.com"
BINANCE_WS = "wss://stream.binance.com:9443/ws/btcusdt@ticker"

MARKET_DATA_DIR = Path("data")
LIVE_RUN_DIR = Path("runs/live_ml")
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
    "predicted_reward_hold",
    "predicted_reward_buy_up",
    "predicted_reward_buy_down",
    "chosen_action",
    "chosen_score",
    "hold_score",
    "score_edge",
    "action_executed",
    "reason",
    "cash_before",
    "up_tokens_before",
    "down_tokens_before",
    "balance_before",
    "cash_after",
    "up_tokens_after",
    "down_tokens_after",
    "balance_after",
    "final_outcome",
    "final_balance",
    "total_reward",
    "reward_to_go",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Live paper-trading ML inference dashboard for BTC 5-minute markets.")
    parser.add_argument("--model", help="Model artifact path. Defaults to newest model in models/.")
    parser.add_argument("--minimum-edge", type=float, default=0.10)
    parser.add_argument("--starting-balance", type=float, default=100.0)
    parser.add_argument("--order-usd", type=float, default=1.0)
    parser.add_argument("--max-orders-per-market", type=int, default=20)
    parser.add_argument("--cooldown-s", type=float, default=5.0)
    parser.add_argument("--max-position-exposure-pct", type=float, default=0.75)
    parser.add_argument("--port", type=int, default=5050)
    return parser.parse_args()


class LiveMLTrader:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.running = True
        self.portfolio = Portfolio(cash=args.starting_balance)
        self.model_bundle = None
        self.model_error = ""
        self.model_path = None
        self.current_market = None
        self.market_file = None
        self.market_writer = None
        self.orders_file = None
        self.orders_writer = None
        self.trajectory_rows = []
        self.recent_orders = []
        self.last_order_time = 0.0
        self.orders_this_market = 0
        self.action_counts = {"hold": 0, "buy_up": 0, "buy_down": 0}

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
            "latest_prediction": {},
            "latest_reason": "starting",
            "last_resolution": "",
        }

        MARKET_DATA_DIR.mkdir(parents=True, exist_ok=True)
        LIVE_RUN_DIR.mkdir(parents=True, exist_ok=True)
        self.load_model(args.model)

    def load_model(self, explicit_path=None):
        try:
            path = Path(explicit_path) if explicit_path else find_latest_reward_model("models")
            if path is None:
                self.model_error = "No model artifact found in models/"
                print(f"MODEL: {self.model_error}")
                return
            self.model_bundle = load_reward_model(path)
            self.model_path = str(path)
            self.model_error = ""
            print(f"MODEL: loaded {path}")
        except Exception as e:
            self.model_bundle = None
            self.model_error = f"{type(e).__name__}: {e}"
            print(f"MODEL ERROR: {self.model_error}")

    def append_market_row(self, row):
        if not self.market_writer or not self.market_file:
            return
        self.market_writer.writerow([format_csv_value(row.get(column)) for column in MARKET_CSV_HEADER])
        self.market_file.flush()

    def begin_market(self, market):
        self.end_market()
        self.current_market = market
        self.portfolio = Portfolio(cash=self.args.starting_balance)
        self.trajectory_rows = []
        self.recent_orders = []
        self.last_order_time = 0.0
        self.orders_this_market = 0
        self.action_counts = {"hold": 0, "buy_up": 0, "buy_down": 0}

        market_path = MARKET_DATA_DIR / f"{market['slug']}.csv"
        self.market_file = market_path.open("w", newline="", encoding="utf-8")
        self.market_writer = csv.writer(self.market_file)
        self.market_writer.writerow(MARKET_CSV_HEADER)

        orders_path = LIVE_RUN_DIR / f"{market['slug']}_orders.csv"
        self.orders_file = orders_path.open("w", newline="", encoding="utf-8")
        self.orders_writer = csv.writer(self.orders_file)
        self.orders_writer.writerow([
            "timestamp",
            "unix_time",
            "action",
            "side",
            "price",
            "usd_amount",
            "tokens",
            "cash_after",
            "up_tokens_after",
            "down_tokens_after",
            "score_edge",
            "reason",
        ])

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
            self.state["latest_prediction"] = {}
            self.state["latest_reason"] = "new market"
            self.state["markets_seen"] += 1

        print()
        print("=" * 72)
        print(f"Market #{self.state['markets_seen']}: {market['question']}")
        print(f"Recording market: {market_path}")
        print(f"Trajectory: {LIVE_RUN_DIR / (market['slug'] + '_trajectory.csv')}")
        print(f"Orders:     {orders_path}")
        print("=" * 72)

    def end_market(self):
        if self.market_file:
            self.market_file.close()
            self.market_file = None
            self.market_writer = None
        if self.orders_file:
            self.orders_file.close()
            self.orders_file = None
            self.orders_writer = None

        if not self.current_market or not self.trajectory_rows:
            return

        final_outcome = self.infer_final_outcome()
        final_balance = self.portfolio.resolve(final_outcome)
        total_reward = final_balance - self.args.starting_balance
        for row in self.trajectory_rows:
            row["final_outcome"] = final_outcome
            row["final_balance"] = final_balance
            row["total_reward"] = total_reward
            row["reward_to_go"] = final_balance - float(row["balance_before"])

        output_path = LIVE_RUN_DIR / f"{self.current_market['slug']}_trajectory.csv"
        with output_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=TRAJECTORY_COLUMNS)
            writer.writeheader()
            writer.writerows(self.trajectory_rows)

        summary = (
            f"{self.current_market['slug']}: outcome={final_outcome} "
            f"final_balance={final_balance:.4f} total_reward={total_reward:.4f} "
            f"holds={self.action_counts['hold']} buy_up={self.action_counts['buy_up']} "
            f"buy_down={self.action_counts['buy_down']}"
        )
        print(summary)
        with self.lock:
            self.state["last_resolution"] = summary

        self.current_market = None

    def infer_final_outcome(self):
        last_btc = None
        last_ptb = None
        with self.lock:
            for _, price in self.state["btc_chainlink"] or self.state["btc_binance"]:
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

    def build_live_tick(self, row):
        return SimpleNamespace(
            timestamp=row["timestamp"],
            unix_time=row["unix_time"],
            elapsed=row["elapsed"],
            seconds_left=row["seconds_left"],
            up_price=row["up_price"],
            down_price=row["down_price"],
            btc_binance=row["btc_binance"],
            btc_chainlink=row["btc_chainlink"],
            price_to_beat=row["price_to_beat"],
        )

    def risk_allows_buy(self, action, portfolio_before):
        if action == "hold":
            return True, ""
        if self.model_bundle is None:
            return False, f"model unavailable: {self.model_error}"
        if time.time() - self.last_order_time < self.args.cooldown_s:
            return False, "cooldown"
        if self.orders_this_market >= self.args.max_orders_per_market:
            return False, "max orders reached"
        if self.portfolio.cash < self.args.order_usd:
            return False, "insufficient simulated cash"
        if portfolio_before["balance_before"] <= 0:
            return False, "non-positive simulated balance"
        projected_exposure = (portfolio_before["position_value"] + self.args.order_usd) / portfolio_before["balance_before"]
        if projected_exposure > self.args.max_position_exposure_pct:
            return False, "max exposure reached"
        return True, ""

    def score_and_trade(self, row):
        if any(row.get(column) is None for column in ["elapsed", "seconds_left", "up_price", "down_price", "price_to_beat"]):
            return self.record_hold(row, "missing market features")
        if row.get("btc_chainlink") is None and row.get("btc_binance") is None:
            return self.record_hold(row, "missing BTC price")
        if self.model_bundle is None:
            return self.record_hold(row, f"model unavailable: {self.model_error}")

        tick = self.build_live_tick(row)
        with self.lock:
            up_history = list(self.state["up_price"])
            down_history = list(self.state["down_price"])
            btc_history = list(self.state["btc_chainlink"] or self.state["btc_binance"])

        history = {
            "up_price": [(elapsed_timestamp(row["unix_time"], row["elapsed"], ts), price) for ts, price in up_history],
            "down_price": [(elapsed_timestamp(row["unix_time"], row["elapsed"], ts), price) for ts, price in down_history],
            "btc_price": [
                (elapsed_timestamp(row["unix_time"], row["elapsed"], ts), price)
                for ts, price in btc_history
            ],
        }
        market_features = build_market_feature_row(tick, history)
        portfolio_before = portfolio_feature_row(self.portfolio, row["up_price"], row["down_price"])
        candidates = build_prediction_rows(market_features, portfolio_before)

        try:
            features = validate_prediction_frame(candidates, self.model_bundle.feature_columns or FEATURE_COLUMNS)
            predictions = self.model_bundle.predict(features)
        except Exception as e:
            return self.record_hold(row, f"feature/model error: {type(e).__name__}: {e}")

        scores = dict(zip(["hold", "buy_up", "buy_down"], predictions))
        hold_score = scores["hold"]
        best_action = max(scores, key=scores.get)
        best_score = scores[best_action]
        edge = best_score - hold_score
        chosen_action = best_action if best_action != "hold" and edge > self.args.minimum_edge else "hold"

        risk_ok, risk_reason = self.risk_allows_buy(chosen_action, portfolio_before)
        if not risk_ok:
            chosen_action = "hold"
            reason = risk_reason
        elif chosen_action == "hold":
            reason = f"edge below threshold {edge:.4f} <= {self.args.minimum_edge:.4f}"
        else:
            reason = f"ml edge {edge:.4f} > {self.args.minimum_edge:.4f}"

        events = execute_action(
            portfolio=self.portfolio,
            action=chosen_action,
            timestamp=row["timestamp"],
            up_price=row["up_price"],
            down_price=row["down_price"],
            usd_amount=self.args.order_usd,
            reason=reason,
        )
        action_executed = chosen_action if chosen_action == "hold" or events else "hold"
        if action_executed != chosen_action:
            reason = f"{reason}; no fill"
        if action_executed != "hold":
            self.last_order_time = time.time()
            self.orders_this_market += len(events)
            self.write_order_events(row, events, edge)
            self.recent_orders.append({
                "time": row["timestamp"],
                "action": action_executed,
                "edge": edge,
                "reason": reason,
            })
            self.recent_orders = self.recent_orders[-20:]
        self.action_counts[action_executed] += 1

        balance_after = self.portfolio.mark_to_market(row["up_price"], row["down_price"])
        trajectory_row = self.make_trajectory_row(
            row, portfolio_before, balance_after, scores, chosen_action, action_executed, reason, edge
        )
        self.trajectory_rows.append(trajectory_row)

        with self.lock:
            self.state["latest_prediction"] = {
                "hold": scores["hold"],
                "buy_up": scores["buy_up"],
                "buy_down": scores["buy_down"],
                "chosen_action": chosen_action,
                "action_executed": action_executed,
                "edge": edge,
            }
            self.state["latest_reason"] = reason

        print(
            f"elapsed={row['elapsed']:.1f} hold={scores['hold']:.4f} "
            f"buy_up={scores['buy_up']:.4f} buy_down={scores['buy_down']:.4f} "
            f"-> {action_executed} edge={edge:.4f}"
        )

    def write_order_events(self, row, events, edge):
        if not self.orders_writer or not self.orders_file:
            return
        for event in events:
            self.orders_writer.writerow([
                event.timestamp,
                row["unix_time"],
                event.action,
                event.side,
                format_csv_value(event.price),
                format_csv_value(event.usd_amount),
                format_csv_value(event.tokens),
                format_csv_value(event.cash_after),
                format_csv_value(event.up_tokens_after),
                format_csv_value(event.down_tokens_after),
                format_csv_value(edge),
                event.reason,
            ])
        self.orders_file.flush()

    def record_hold(self, row, reason):
        scores = {"hold": 0.0, "buy_up": 0.0, "buy_down": 0.0}
        portfolio_before = portfolio_feature_row(self.portfolio, row.get("up_price") or 0.0, row.get("down_price") or 0.0)
        balance_after = portfolio_before["balance_before"]
        self.action_counts["hold"] += 1
        self.trajectory_rows.append(
            self.make_trajectory_row(row, portfolio_before, balance_after, scores, "hold", "hold", reason, 0.0)
        )
        with self.lock:
            self.state["latest_prediction"] = {
                "hold": 0.0,
                "buy_up": 0.0,
                "buy_down": 0.0,
                "chosen_action": "hold",
                "action_executed": "hold",
                "edge": 0.0,
            }
            self.state["latest_reason"] = reason

    def make_trajectory_row(self, row, portfolio_before, balance_after, scores, chosen_action, action_executed, reason, edge):
        return {
            "timestamp": row.get("timestamp"),
            "unix_time": row.get("unix_time"),
            "elapsed": row.get("elapsed"),
            "seconds_left": row.get("seconds_left"),
            "up_price": row.get("up_price"),
            "down_price": row.get("down_price"),
            "btc_price": row.get("btc_chainlink") if row.get("btc_chainlink") is not None else row.get("btc_binance"),
            "price_to_beat": row.get("price_to_beat"),
            "predicted_reward_hold": scores["hold"],
            "predicted_reward_buy_up": scores["buy_up"],
            "predicted_reward_buy_down": scores["buy_down"],
            "chosen_action": chosen_action,
            "chosen_score": scores[chosen_action],
            "hold_score": scores["hold"],
            "score_edge": edge,
            "action_executed": action_executed,
            "reason": reason,
            "cash_before": portfolio_before["cash_before"],
            "up_tokens_before": portfolio_before["up_tokens_before"],
            "down_tokens_before": portfolio_before["down_tokens_before"],
            "balance_before": portfolio_before["balance_before"],
            "cash_after": self.portfolio.cash,
            "up_tokens_after": self.portfolio.up_tokens,
            "down_tokens_after": self.portfolio.down_tokens,
            "balance_after": balance_after,
            "final_outcome": "",
            "final_balance": "",
            "total_reward": "",
            "reward_to_go": "",
        }


def elapsed_timestamp(now_unix, now_elapsed, series_unix_ts):
    return now_elapsed - (now_unix - series_unix_ts)


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


def poll_clob_loop(trader):
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
                and now_t >= trader.state["market_start"]
            ):
                trader.state["price_to_beat"] = trader.state["btc_chainlink"][-1][1]

        row = trader.latest_snapshot()
        trader.append_market_row(row)
        trader.score_and_trade(row)

        if remaining <= 20:
            sleep_s = 0.05
        elif remaining <= 60:
            sleep_s = 0.08
        else:
            sleep_s = 0.10
        time.sleep(sleep_s)


def run_binance(trader):
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
                                with trader.lock:
                                    trader.state["btc_binance"].append((time.time(), float(data["c"])))
            except Exception as e:
                print(f"Binance error: {e}")
                await asyncio.sleep(5)

    asyncio.run(connect())


def run_rtds(trader):
    async def connect():
        while trader.running:
            try:
                async with websockets.connect(RTDS_URL) as ws:
                    await ws.send(json.dumps({
                        "action": "subscribe",
                        "subscriptions": [{"topic": "crypto_prices_chainlink", "type": "*", "filters": ""}],
                    }))
                    print("Chainlink/RTDS connected")
                    async for msg in ws:
                        if not trader.running:
                            break
                        if msg == "PONG" or not msg.strip():
                            continue
                        try:
                            data = json.loads(msg)
                        except Exception:
                            continue
                        payload = data.get("payload", {})
                        if payload.get("symbol") == "btc/usd" and payload.get("value") is not None:
                            with trader.lock:
                                trader.state["btc_chainlink"].append((time.time(), float(payload["value"])))
            except Exception as e:
                print(f"RTDS error: {e}")
                await asyncio.sleep(5)

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
  <title>Live ML BTC Trader</title>
  <script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
  <style>
    body { margin:0; font-family: Consolas, monospace; background:#090b0f; color:#e6e8ee; }
    .top { display:flex; justify-content:space-between; padding:14px 18px; background:#121722; border-bottom:1px solid #253044; }
    .title { color:#7fb3ff; font-weight:700; }
    .grid { display:grid; grid-template-columns: repeat(4, 1fr); gap:1px; background:#253044; }
    .cell { background:#0f141d; padding:14px 16px; min-height:62px; }
    .label { color:#7d8797; font-size:11px; text-transform:uppercase; }
    .value { font-size:20px; margin-top:5px; }
    .wide { grid-column: span 2; }
    .orders { padding:16px; }
    table { width:100%; border-collapse:collapse; font-size:12px; }
    td, th { border-bottom:1px solid #202a3a; padding:7px; text-align:left; }
    .green { color:#4ade80; } .red { color:#f87171; } .blue { color:#60a5fa; }
  </style>
</head>
<body>
  <div class="top">
    <div class="title">LIVE ML BTC PAPER TRADER</div>
    <div id="slug">waiting</div>
  </div>
  <div class="grid">
    <div class="cell wide"><div class="label">Question</div><div class="value" id="question">--</div></div>
    <div class="cell"><div class="label">Countdown</div><div class="value" id="countdown">--</div></div>
    <div class="cell"><div class="label">Model</div><div class="value" id="model" style="font-size:12px">--</div></div>
    <div class="cell"><div class="label">UP / DOWN</div><div class="value" id="prices">--</div></div>
    <div class="cell"><div class="label">BTC</div><div class="value" id="btc">--</div></div>
    <div class="cell"><div class="label">Price to beat</div><div class="value" id="ptb">--</div></div>
    <div class="cell"><div class="label">Distance</div><div class="value" id="distance">--</div></div>
    <div class="cell"><div class="label">Balance</div><div class="value" id="balance">--</div></div>
    <div class="cell"><div class="label">Hold score</div><div class="value" id="hold">--</div></div>
    <div class="cell"><div class="label">Buy UP score</div><div class="value green" id="buyup">--</div></div>
    <div class="cell"><div class="label">Buy DOWN score</div><div class="value red" id="buydown">--</div></div>
    <div class="cell"><div class="label">Chosen / Edge</div><div class="value" id="chosen">--</div></div>
    <div class="cell"><div class="label">Position</div><div class="value" id="position">--</div></div>
    <div class="cell wide"><div class="label">Reason</div><div class="value" id="reason" style="font-size:13px">--</div></div>
    <div class="cell"><div class="label">Last resolution</div><div class="value" id="resolution" style="font-size:12px">--</div></div>
  </div>
  <div class="orders">
    <h3>Recent simulated orders</h3>
    <table><thead><tr><th>Time</th><th>Action</th><th>Edge</th><th>Reason</th></tr></thead><tbody id="orders"></tbody></table>
  </div>
<script>
const socket = io();
function money(v) { return v == null ? '--' : '$' + Number(v).toFixed(2); }
function num(v, d=4) { return v == null ? '--' : Number(v).toFixed(d); }
socket.on('tick', d => {
  document.getElementById('slug').textContent = d.slug || 'waiting';
  document.getElementById('question').textContent = d.question || '--';
  document.getElementById('countdown').textContent = d.remaining == null ? '--' : d.remaining.toFixed(0) + 's';
  document.getElementById('model').textContent = d.model_path || d.model_error || '--';
  document.getElementById('prices').innerHTML = '<span class="green">' + num(d.up_price,3) + '</span> / <span class="red">' + num(d.down_price,3) + '</span>';
  document.getElementById('btc').textContent = money(d.btc_price);
  document.getElementById('ptb').textContent = money(d.price_to_beat);
  document.getElementById('distance').textContent = d.distance_pct == null ? '--' : (d.distance_pct * 100).toFixed(4) + '%';
  document.getElementById('balance').textContent = money(d.balance);
  document.getElementById('hold').textContent = num(d.pred.hold);
  document.getElementById('buyup').textContent = num(d.pred.buy_up);
  document.getElementById('buydown').textContent = num(d.pred.buy_down);
  document.getElementById('chosen').textContent = (d.pred.action_executed || '--') + ' / ' + num(d.pred.edge);
  document.getElementById('position').textContent = 'UP ' + num(d.up_tokens,2) + ' / DOWN ' + num(d.down_tokens,2);
  document.getElementById('reason').textContent = d.reason || '--';
  document.getElementById('resolution').textContent = d.last_resolution || '--';
  document.getElementById('orders').innerHTML = d.orders.map(o =>
    '<tr><td>'+o.time+'</td><td>'+o.action+'</td><td>'+num(o.edge)+'</td><td>'+o.reason+'</td></tr>'
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
        with TRADER.lock:
            up_price = TRADER.state["up_price"][-1][1] if TRADER.state["up_price"] else None
            down_price = TRADER.state["down_price"][-1][1] if TRADER.state["down_price"] else None
            btc_price = (
                TRADER.state["btc_chainlink"][-1][1]
                if TRADER.state["btc_chainlink"]
                else (TRADER.state["btc_binance"][-1][1] if TRADER.state["btc_binance"] else None)
            )
            ptb = TRADER.state["price_to_beat"]
            remaining = TRADER.state["market_end"] - time.time() if TRADER.state["market_end"] else None
            pred = dict(TRADER.state["latest_prediction"])
            reason = TRADER.state["latest_reason"]
            payload = {
                "slug": TRADER.state["current_slug"],
                "question": TRADER.state["market_question"],
                "remaining": remaining,
                "up_price": up_price,
                "down_price": down_price,
                "btc_price": btc_price,
                "price_to_beat": ptb,
                "distance_pct": ((btc_price - ptb) / ptb) if btc_price is not None and ptb else None,
                "balance": TRADER.portfolio.mark_to_market(up_price or 0.0, down_price or 0.0),
                "up_tokens": TRADER.portfolio.up_tokens,
                "down_tokens": TRADER.portfolio.down_tokens,
                "pred": pred,
                "reason": reason,
                "orders": list(TRADER.recent_orders),
                "model_path": TRADER.model_path,
                "model_error": TRADER.model_error,
                "last_resolution": TRADER.state["last_resolution"],
            }
        socketio.emit("tick", payload)
        time.sleep(0.5)


def main():
    global TRADER
    args = parse_args()
    TRADER = LiveMLTrader(args)

    print("Live ML BTC paper trader")
    print("=" * 50)
    print(f"Dashboard: http://localhost:{args.port}")
    print(f"Model:     {TRADER.model_path or TRADER.model_error}")
    print(f"Data:      {MARKET_DATA_DIR}/<slug>.csv")
    print(f"Runs:      {LIVE_RUN_DIR}/<slug>_trajectory.csv")
    print("Mode:      paper trading only")
    print("=" * 50)

    threading.Thread(target=poll_clob_loop, args=(TRADER,), daemon=True).start()
    threading.Thread(target=run_binance, args=(TRADER,), daemon=True).start()
    threading.Thread(target=run_rtds, args=(TRADER,), daemon=True).start()
    threading.Thread(target=emit_tick_loop, daemon=True).start()

    socketio.run(app, host="0.0.0.0", port=args.port, debug=False, allow_unsafe_werkzeug=True)


if __name__ == "__main__":
    main()
