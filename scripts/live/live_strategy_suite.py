import argparse
import asyncio
import csv
import json
import re
import sys
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import aiohttp
import requests

try:
    import websockets
except ModuleNotFoundError:
    websockets = None

try:
    from flask import Flask, render_template_string
    from flask_socketio import SocketIO
except ModuleNotFoundError:
    Flask = None
    SocketIO = None
    render_template_string = None

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.config_loader import build_strategy_from_config, load_yaml
from simulator.execution import execute_action
from simulator.batch import resolve_effective_market_balance
from simulator.models import DecisionState, MarketTick
from simulator.portfolio import Portfolio
from simulator.strategies import StrategyDecision


GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
RTDS_URL = "wss://ws-live-data.polymarket.com"
BINANCE_WS = "wss://stream.binance.com:9443/ws/btcusdt@ticker"

RUN_ROOT = Path("runs/live_strategy_suite")
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
    "price_to_beat_source",
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
    "price_to_beat_source",
    "market_slug",
    "strategy_name",
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
    "master_balance_before_market",
    "market_simulated_starting_balance",
    "market_simulated_final_balance",
    "master_balance_after_market",
    "final_outcome",
    "total_reward",
]
SUMMARY_COLUMNS = [
    "timestamp",
    "status",
    "market_file",
    "market_slug",
    "strategy_name",
    "strategy_config",
    "final_outcome",
    "price_to_beat",
    "price_to_beat_source",
    "final_balance",
    "starting_balance",
    "master_balance_before_market",
    "market_simulated_starting_balance",
    "master_balance_after_market",
    "total_reward",
    "rows_written",
    "orders_placed",
    "buy_up_events",
    "buy_down_events",
    "sell_up_events",
    "sell_down_events",
    "output_csv",
]


def parse_args():
    parser = argparse.ArgumentParser(description="Run a folder of rule strategies on live Polymarket market data.")
    parser.add_argument(
        "--strategy-folder",
        default="configs/strategies/family_fixed100",
        help="Folder containing strategy YAML files.",
    )
    parser.add_argument("--strategy-pattern", default="*.yaml", help="Glob pattern inside the strategy folder.")
    parser.add_argument("--starting-balance", type=float, default=100.0, help="Paper balance per strategy per market.")
    parser.add_argument(
        "--balance-config",
        default="",
        help="Optional batch-style config with starting_balance and effective_market_balance_bands for compounded live paper balances.",
    )
    parser.add_argument("--run-id", default="", help="Optional run folder name under runs/live_strategy_suite.")
    parser.add_argument("--port", type=int, default=5052, help="Local dashboard port.")
    parser.add_argument("--max-markets", type=int, default=0, help="Stop after this many completed markets. 0 means run forever.")
    parser.add_argument(
        "--trajectory-log-mode",
        choices=["all", "actions"],
        default="all",
        help="Write every strategy tick to trajectory CSVs, or only rows where an action/event occurred.",
    )
    parser.add_argument(
        "--max-target-fallback-elapsed",
        type=float,
        default=5.0,
        help="Only infer price_to_beat from live BTC if this many seconds or less have elapsed in the market.",
    )
    return parser.parse_args()


def slugify(value):
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value).strip())
    return value.strip("_") or "strategy"


def format_csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.10g}"
    return value


def parse_numeric(value):
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        cleaned = value.strip().replace("$", "").replace(",", "")
        try:
            return float(cleaned)
        except ValueError:
            return None
    return None


def extract_price_to_beat(payload):
    key_hints = {
        "priceToBeat",
        "price_to_beat",
        "targetPrice",
        "target_price",
        "strikePrice",
        "strike_price",
        "initialPrice",
        "initial_price",
    }
    text_fields = []

    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in key_hints:
                    numeric = parse_numeric(item)
                    if numeric and numeric > 1000:
                        return numeric
                if isinstance(item, str) and key.lower() in {"title", "question", "description", "rules", "resolution"}:
                    text_fields.append(item)
                found = walk(item)
                if found is not None:
                    return found
        elif isinstance(value, list):
            for item in value:
                found = walk(item)
                if found is not None:
                    return found
        return None

    found = walk(payload)
    if found is not None:
        return found

    for text in text_fields:
        matches = re.findall(r"\$?\b(\d{2,3}(?:,\d{3})+(?:\.\d+)?|\d{5,6}(?:\.\d+)?)\b", text)
        for match in matches:
            numeric = parse_numeric(match)
            if numeric and numeric > 1000:
                return numeric
    return None


def extract_chainlink_btc_price(payload):
    if isinstance(payload, list):
        for item in payload:
            price = extract_chainlink_btc_price(item)
            if price is not None:
                return price
        return None

    if not isinstance(payload, dict):
        return None

    asset_text = " ".join(
        str(payload.get(key, ""))
        for key in ["asset", "symbol", "ticker", "base", "pair"]
    ).upper()
    if "BTC" in asset_text:
        for key in ["price", "value", "answer", "rate"]:
            price = parse_numeric(payload.get(key))
            if price and price > 1000:
                return price

    for key in ["payload", "data", "event", "message"]:
        price = extract_chainlink_btc_price(payload.get(key))
        if price is not None:
            return price

    return None


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
            price_to_beat = extract_price_to_beat({"event": ev, "market": m})
            if end > now:
                return {
                    "slug": slug,
                    "slot": s,
                    "question": m["question"],
                    "up_token": tids[0],
                    "down_token": tids[1],
                    "start_time": start,
                    "end_time": end,
                    "price_to_beat": price_to_beat,
                    "price_to_beat_source": "gamma" if price_to_beat is not None else "",
                }
    return None


@dataclass
class StrategyRuntime:
    config_path: Path
    config: dict
    starting_balance: float
    master_balance: float = 0.0
    master_balance_before_market: float = 0.0
    market_starting_balance: float = 0.0
    name: str = ""
    strategy: object = None
    portfolio: Portfolio = None
    market_spend_used: float = 0.0
    last_action: str = "none"
    orders_placed: int = 0
    event_counts: dict = field(default_factory=lambda: {"buy_up": 0, "buy_down": 0, "sell_up": 0, "sell_down": 0})
    rows: list[dict] = field(default_factory=list)

    def __post_init__(self):
        self.name = self.config.get("name") or self.config_path.stem
        self.master_balance = self.starting_balance
        self.master_balance_before_market = self.starting_balance
        self.market_starting_balance = self.starting_balance

    def reset_for_market(self, market_starting_balance: float | None = None):
        self.master_balance_before_market = self.master_balance
        self.market_starting_balance = (
            float(market_starting_balance)
            if market_starting_balance is not None
            else self.starting_balance
        )
        self.strategy = build_strategy_from_config(self.config)
        self.portfolio = Portfolio(cash=self.market_starting_balance)
        self.market_spend_used = 0.0
        self.last_action = "none"
        self.orders_placed = 0
        self.event_counts = {"buy_up": 0, "buy_down": 0, "sell_up": 0, "sell_down": 0}
        self.rows = []


class LiveStrategySuite:
    def __init__(self, args):
        self.args = args
        self.lock = threading.Lock()
        self.running = True
        self.completed_markets = 0
        self.strategy_folder = Path(args.strategy_folder)
        self.balance_cfg = load_yaml(args.balance_config) if args.balance_config else {}
        self.compound_balance = bool(args.balance_config)
        self.base_starting_balance = float(
            self.balance_cfg.get("starting_balance", args.starting_balance)
            if self.compound_balance
            else args.starting_balance
        )
        self.strategies = self.load_strategies()

        run_id = args.run_id or f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{slugify(self.strategy_folder.name)}"
        self.run_dir = RUN_ROOT / run_id
        self.market_data_dir = self.run_dir / "market_data"
        self.trajectory_dir = self.run_dir / "trajectories"
        self.summary_path = self.run_dir / "summary.csv"
        self.latest_json_path = self.run_dir / "latest.json"
        self.summary_html_path = self.run_dir / "live_summary.html"
        self.market_data_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.ensure_summary_header()

        self.state = {
            "btc_binance": [],
            "btc_chainlink": [],
            "up_price": [],
            "down_price": [],
            "price_to_beat": None,
            "price_to_beat_source": "",
            "target_fallback_warning_logged": False,
            "market_question": "",
            "current_slug": "",
            "market_start": 0.0,
            "market_end": 0.0,
            "markets_seen": 0,
        }
        self.current_market = None
        self.market_file = None
        self.market_writer = None
        self.latest_non_hold_actions = []
        self.recent_completed = []

    def load_strategies(self):
        if not self.strategy_folder.exists():
            raise FileNotFoundError(f"Strategy folder not found: {self.strategy_folder}")
        paths = sorted(self.strategy_folder.glob(self.args.strategy_pattern))
        if not paths:
            raise FileNotFoundError(f"No strategy files matched {self.args.strategy_pattern!r} in {self.strategy_folder}")
        return [
            StrategyRuntime(path, load_yaml(path), self.base_starting_balance)
            for path in paths
        ]

    def ensure_summary_header(self):
        if self.summary_path.exists():
            return
        with self.summary_path.open("w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(SUMMARY_COLUMNS)

    def begin_market(self, market):
        self.end_market()
        self.current_market = market
        for runtime in self.strategies:
            market_starting_balance = runtime.starting_balance
            if self.compound_balance:
                effective_balance = resolve_effective_market_balance(runtime.master_balance, self.balance_cfg)
                market_starting_balance = effective_balance if effective_balance is not None else runtime.master_balance
            runtime.reset_for_market(market_starting_balance)

        market_path = self.market_data_dir / f"{market['slug']}.csv"
        self.market_file = market_path.open("w", newline="", encoding="utf-8")
        self.market_writer = csv.writer(self.market_file)
        self.market_writer.writerow(MARKET_CSV_HEADER)

        with self.lock:
            self.state["current_slug"] = market["slug"]
            self.state["market_question"] = market["question"]
            self.state["price_to_beat"] = market.get("price_to_beat")
            self.state["price_to_beat_source"] = market.get("price_to_beat_source", "")
            self.state["target_fallback_warning_logged"] = False
            self.state["market_start"] = market["start_time"]
            self.state["market_end"] = market["end_time"]
            self.state["up_price"].clear()
            self.state["down_price"].clear()
            self.state["btc_binance"].clear()
            self.state["btc_chainlink"].clear()
            self.state["markets_seen"] += 1
            self.latest_non_hold_actions.clear()

        print()
        print("=" * 80)
        print(f"Market #{self.state['markets_seen']}: {market['question']}")
        print(f"Strategies: {len(self.strategies)} from {self.strategy_folder}")
        if self.compound_balance:
            active = [runtime.market_starting_balance for runtime in self.strategies]
            print(f"Compound balance config: {self.args.balance_config}")
            print(
                f"Master balance range: {min(runtime.master_balance_before_market for runtime in self.strategies):.2f}"
                f"-{max(runtime.master_balance_before_market for runtime in self.strategies):.2f}"
            )
            print(f"Effective market balance range: {min(active):.2f}-{max(active):.2f}")
        else:
            print(f"Paper balance per strategy: {self.args.starting_balance:.2f}")
        if market.get("price_to_beat") is not None:
            print(f"Price to beat: {market['price_to_beat']:.2f} ({market.get('price_to_beat_source')})")
        else:
            print("Price to beat: waiting for live BTC feed fallback")
        print(f"Run dir: {self.run_dir}")
        print("=" * 80)

    def end_market(self):
        if self.market_file:
            self.market_file.close()
            self.market_file = None
            self.market_writer = None

        if not self.current_market:
            return

        if not any(runtime.rows for runtime in self.strategies):
            self.current_market = None
            return

        final_outcome = self.infer_final_outcome()
        market_slug = self.current_market["slug"]
        market_file = str(self.market_data_dir / f"{market_slug}.csv")
        completed_rows = []

        for runtime in self.strategies:
            final_balance = runtime.portfolio.resolve(final_outcome) if runtime.portfolio else runtime.market_starting_balance
            total_reward = final_balance - runtime.market_starting_balance
            if self.compound_balance:
                runtime.master_balance = max(0.0, runtime.master_balance_before_market + total_reward)

            for row in runtime.rows:
                row["market_simulated_final_balance"] = final_balance
                row["master_balance_after_market"] = runtime.master_balance
                row["final_outcome"] = final_outcome
                row["total_reward"] = total_reward

            strategy_dir = self.trajectory_dir / slugify(runtime.name)
            strategy_dir.mkdir(parents=True, exist_ok=True)
            output_path = strategy_dir / f"{market_slug}.csv"
            rows_to_write = runtime.rows
            if self.args.trajectory_log_mode == "actions":
                rows_to_write = [
                    row
                    for row in runtime.rows
                    if row.get("action") != "hold" or int(row.get("events_count") or 0) > 0
                ]
            with output_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=TRAJECTORY_COLUMNS)
                writer.writeheader()
                writer.writerows(rows_to_write)

            summary_row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "status": "ok",
                "market_file": market_file,
                "market_slug": market_slug,
                "strategy_name": runtime.name,
                "strategy_config": str(runtime.config_path),
                "final_outcome": final_outcome,
                "price_to_beat": runtime.rows[-1].get("price_to_beat", "") if runtime.rows else "",
                "price_to_beat_source": runtime.rows[-1].get("price_to_beat_source", "") if runtime.rows else "",
                "final_balance": final_balance,
                "starting_balance": runtime.starting_balance,
                "master_balance_before_market": runtime.master_balance_before_market,
                "market_simulated_starting_balance": runtime.market_starting_balance,
                "master_balance_after_market": runtime.master_balance,
                "total_reward": total_reward,
                "rows_written": len(rows_to_write),
                "orders_placed": runtime.orders_placed,
                "buy_up_events": runtime.event_counts["buy_up"],
                "buy_down_events": runtime.event_counts["buy_down"],
                "sell_up_events": runtime.event_counts["sell_up"],
                "sell_down_events": runtime.event_counts["sell_down"],
                "output_csv": str(output_path),
            }
            completed_rows.append(summary_row)

        with self.summary_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS)
            writer.writerows(completed_rows)

        completed_rows.sort(key=lambda row: float(row["total_reward"]), reverse=True)
        self.completed_markets += 1
        self.recent_completed.insert(0, {
            "market_slug": market_slug,
            "final_outcome": final_outcome,
            "top_strategy": completed_rows[0]["strategy_name"],
            "top_reward": completed_rows[0]["total_reward"],
            "bottom_strategy": completed_rows[-1]["strategy_name"],
            "bottom_reward": completed_rows[-1]["total_reward"],
        })
        self.recent_completed = self.recent_completed[:20]
        self.write_static_summary()

        print(f"{market_slug}: outcome={final_outcome} top={completed_rows[0]['strategy_name']} reward={completed_rows[0]['total_reward']:.4f}")
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
            price_to_beat_source = self.state["price_to_beat_source"]
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
            "price_to_beat_source": price_to_beat_source,
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

    def step_strategies(self, row):
        tick = self.build_tick(row)
        market_slug = self.current_market["slug"] if self.current_market else ""
        for runtime in self.strategies:
            current_balance = runtime.portfolio.mark_to_market(row["up_price"], row["down_price"])
            state = DecisionState(
                tick=tick,
                cash=runtime.portfolio.cash,
                up_tokens=runtime.portfolio.up_tokens,
                down_tokens=runtime.portfolio.down_tokens,
                current_balance=current_balance,
                market_start_balance=runtime.market_starting_balance,
                market_spend_used=runtime.market_spend_used,
                last_action=runtime.last_action,
                orders_placed=runtime.orders_placed,
            )
            if tick.seconds_left is not None and tick.seconds_left <= 0:
                decision = StrategyDecision("hold", "market closed")
            else:
                decision = runtime.strategy.decide(state)
            usd_amount = decision.usd_amount if decision.usd_amount is not None else float(runtime.config.get("order_usd", 1.0))
            spend_before = runtime.market_spend_used
            events = execute_action(
                portfolio=runtime.portfolio,
                action=decision.action,
                timestamp=row["timestamp"],
                up_price=row["up_price"],
                down_price=row["down_price"],
                usd_amount=usd_amount,
                reason=decision.reason,
            )
            for event in events:
                key = f"{event.action}_{event.side}"
                if key in runtime.event_counts:
                    runtime.event_counts[key] += 1
            if events:
                runtime.orders_placed += len(events)
                runtime.market_spend_used += sum(event.usd_amount for event in events if event.action == "buy")

            balance_after = runtime.portfolio.mark_to_market(row["up_price"], row["down_price"])
            runtime.rows.append({
                "timestamp": row["timestamp"],
                "unix_time": row["unix_time"],
                "elapsed": row["elapsed"],
                "seconds_left": row["seconds_left"],
                "up_price": row["up_price"],
                "down_price": row["down_price"],
                "btc_price": row["btc_chainlink"] if row["btc_chainlink"] is not None else row["btc_binance"],
                "price_to_beat": row["price_to_beat"],
                "price_to_beat_source": row["price_to_beat_source"],
                "market_slug": market_slug,
                "strategy_name": runtime.name,
                "cash_before": state.cash,
                "up_tokens_before": state.up_tokens,
                "down_tokens_before": state.down_tokens,
                "balance_before": state.current_balance,
                "action": decision.action,
                "reason": decision.reason,
                "usd_amount": usd_amount,
                "events_count": len(events),
                "market_spend_used_before": spend_before,
                "market_spend_used_after": runtime.market_spend_used,
                "cash_after": runtime.portfolio.cash,
                "up_tokens_after": runtime.portfolio.up_tokens,
                "down_tokens_after": runtime.portfolio.down_tokens,
                "balance_after": balance_after,
                "master_balance_before_market": runtime.master_balance_before_market,
                "market_simulated_starting_balance": runtime.market_starting_balance,
                "market_simulated_final_balance": "",
                "master_balance_after_market": "",
                "final_outcome": "",
                "total_reward": "",
            })
            runtime.last_action = decision.action

            if decision.action != "hold" or events:
                action_row = {
                    "timestamp": row["timestamp"],
                    "strategy_name": runtime.name,
                    "action": decision.action,
                    "reason": decision.reason,
                    "usd_amount": usd_amount,
                    "events_count": len(events),
                    "balance_after": balance_after,
                }
                with self.lock:
                    self.latest_non_hold_actions.insert(0, action_row)
                    self.latest_non_hold_actions = self.latest_non_hold_actions[:30]
                print(
                    f"{row['timestamp']} {runtime.name}: {decision.action} "
                    f"usd={usd_amount:.2f} events={len(events)} reason={decision.reason}"
                )

    def dashboard_payload(self):
        with self.lock:
            up_price = self.state["up_price"][-1][1] if self.state["up_price"] else None
            down_price = self.state["down_price"][-1][1] if self.state["down_price"] else None
            btc_price = self.state["btc_chainlink"][-1][1] if self.state["btc_chainlink"] else (
                self.state["btc_binance"][-1][1] if self.state["btc_binance"] else None
            )
            payload = {
                "market_question": self.state["market_question"],
                "current_slug": self.state["current_slug"],
                "markets_seen": self.state["markets_seen"],
                "seconds_left": self.state["market_end"] - time.time() if self.state["market_end"] else None,
                "elapsed": time.time() - self.state["market_start"] if self.state["market_start"] else None,
                "up_price": up_price,
                "down_price": down_price,
                "btc_price": btc_price,
                "price_to_beat": self.state["price_to_beat"],
                "price_to_beat_source": self.state["price_to_beat_source"],
                "latest_actions": list(self.latest_non_hold_actions),
            }
        payload["strategy_count"] = len(self.strategies)
        payload["run_dir"] = str(self.run_dir)
        payload["summary_csv"] = str(self.summary_path)
        payload["summary_html"] = str(self.summary_html_path)
        payload["recent_completed"] = list(self.recent_completed)
        payload["leaderboard"] = self.current_leaderboard(up_price, down_price)
        payload["completed_summary"] = self.summary_snapshot()
        distance = None
        if btc_price is not None and payload["price_to_beat"] is not None:
            distance = btc_price - payload["price_to_beat"]
        payload["distance"] = distance
        return payload

    def current_leaderboard(self, up_price, down_price):
        rows = []
        for runtime in self.strategies:
            if not runtime.portfolio:
                balance = runtime.market_starting_balance
                cash = runtime.market_starting_balance
                up_tokens = 0.0
                down_tokens = 0.0
            elif up_price is None or down_price is None:
                balance = runtime.portfolio.cash
                cash = runtime.portfolio.cash
                up_tokens = runtime.portfolio.up_tokens
                down_tokens = runtime.portfolio.down_tokens
            else:
                balance = runtime.portfolio.mark_to_market(up_price, down_price)
                cash = runtime.portfolio.cash
                up_tokens = runtime.portfolio.up_tokens
                down_tokens = runtime.portfolio.down_tokens
            rows.append({
                "strategy_name": runtime.name,
                "balance": balance,
                "reward": balance - runtime.market_starting_balance,
                "master_balance": runtime.master_balance,
                "master_balance_before_market": runtime.master_balance_before_market,
                "market_starting_balance": runtime.market_starting_balance,
                "cash": cash,
                "up_tokens": up_tokens,
                "down_tokens": down_tokens,
                "orders": runtime.orders_placed,
                "last_action": runtime.last_action,
            })
        rows.sort(key=lambda item: item["reward"], reverse=True)
        return rows

    def write_static_summary(self):
        summary = self.summary_snapshot()
        html = render_static_summary(self.run_dir, self.summary_path, summary)
        self.summary_html_path.write_text(html, encoding="utf-8")
        self.latest_json_path.write_text(json.dumps(self.dashboard_payload(), indent=2), encoding="utf-8")

    def summary_snapshot(self):
        rows = []
        if self.summary_path.exists():
            with self.summary_path.open("r", newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
        return build_completed_summary(rows)


def poll_clob_loop(suite: LiveStrategySuite):
    current_market = None
    while suite.running:
        now = time.time()
        if current_market is None or now >= current_market["end_time"]:
            if current_market is not None:
                suite.end_market()
                if suite.args.max_markets and suite.completed_markets >= suite.args.max_markets:
                    suite.running = False
                    break
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
            suite.begin_market(current_market)
            wait = current_market["start_time"] - time.time()
            if wait > 0:
                time.sleep(wait)

        remaining = current_market["end_time"] - time.time()
        if remaining <= 0:
            continue

        try:
            resp = requests.get(
                f"{CLOB_API}/price",
                params={"token_id": current_market["up_token"], "side": "buy"},
                timeout=3,
            )
            up_price = float(resp.json().get("price", 0))
            down_price = round(1 - up_price, 3)
        except Exception as exc:
            print(f"CLOB price error: {exc}")
            time.sleep(0.2)
            continue

        now_t = time.time()
        with suite.lock:
            suite.state["up_price"].append((now_t, up_price))
            suite.state["down_price"].append((now_t, down_price))
            elapsed = now_t - suite.state["market_start"] if suite.state["market_start"] else None
            fallback_allowed = elapsed is not None and elapsed <= suite.args.max_target_fallback_elapsed
            if suite.state["price_to_beat"] is None and fallback_allowed and suite.state["btc_chainlink"]:
                suite.state["price_to_beat"] = suite.state["btc_chainlink"][-1][1]
                suite.state["price_to_beat_source"] = "chainlink_start_fallback"
                print(f"Price to beat fallback set from Chainlink at elapsed={elapsed:.3f}s: {suite.state['price_to_beat']:.2f}")
            if suite.state["price_to_beat"] is None and fallback_allowed and suite.state["btc_binance"]:
                suite.state["price_to_beat"] = suite.state["btc_binance"][-1][1]
                suite.state["price_to_beat_source"] = "binance_start_fallback"
                print(f"Price to beat fallback set from Binance at elapsed={elapsed:.3f}s: {suite.state['price_to_beat']:.2f}")
            if (
                suite.state["price_to_beat"] is None
                and elapsed is not None
                and elapsed > suite.args.max_target_fallback_elapsed
                and not suite.state["target_fallback_warning_logged"]
            ):
                suite.state["target_fallback_warning_logged"] = True
                print(
                    f"Skipping strategy decisions for {suite.state['current_slug']}: "
                    f"missing price_to_beat after elapsed={elapsed:.3f}s. "
                    "Waiting for next full market."
                )

        row = suite.latest_snapshot()
        suite.append_market_row(row)
        if all(row.get(column) is not None for column in ["elapsed", "seconds_left", "up_price", "down_price", "price_to_beat"]) and (
            row.get("btc_chainlink") is not None or row.get("btc_binance") is not None
        ):
            suite.step_strategies(row)

        if remaining <= 20:
            sleep_s = 0.05
        elif remaining <= 60:
            sleep_s = 0.08
        else:
            sleep_s = 0.10
        time.sleep(sleep_s)


def start_binance_loop(suite: LiveStrategySuite):
    async def connect():
        while suite.running:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.ws_connect(BINANCE_WS) as ws:
                        print("Binance connected")
                        async for msg in ws:
                            if not suite.running:
                                break
                            if msg.type == aiohttp.WSMsgType.TEXT:
                                data = json.loads(msg.data)
                                price = float(data["c"])
                                with suite.lock:
                                    suite.state["btc_binance"].append((time.time(), price))
            except Exception as exc:
                print(f"Binance reconnecting: {exc}")
                await asyncio.sleep(2)

    asyncio.run(connect())


def start_chainlink_loop(suite: LiveStrategySuite):
    if websockets is None:
        print("Chainlink feed disabled: install the 'websockets' package to enable it.")
        return

    async def connect():
        while suite.running:
            try:
                async with websockets.connect(RTDS_URL) as ws:
                    await ws.send(json.dumps({
                        "action": "subscribe",
                        "subscriptions": [{"topic": "crypto_prices_chainlink", "type": "*", "filters": ""}],
                    }))
                    print("Chainlink connected")
                    async for msg in ws:
                        if not suite.running:
                            break
                        if msg == "PONG" or not msg.strip():
                            continue
                        try:
                            payload = json.loads(msg)
                        except json.JSONDecodeError:
                            continue
                        price = extract_chainlink_btc_price(payload)
                        if price is None:
                            continue
                        with suite.lock:
                            suite.state["btc_chainlink"].append((time.time(), price))
            except Exception as exc:
                print(f"Chainlink reconnecting: {exc}")
                await asyncio.sleep(2)

    asyncio.run(connect())


def to_float(value, default=0.0):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def to_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def build_completed_summary(rows):
    ok_rows = [row for row in rows if row.get("status", "ok") == "ok"]
    leaderboard = {}
    market_order = []
    market_seen = set()
    market_rewards = {}

    for row in ok_rows:
        market_slug = row.get("market_slug") or Path(row.get("market_file", "")).stem
        if market_slug and market_slug not in market_seen:
            market_seen.add(market_slug)
            market_order.append(market_slug)
        name = row.get("strategy_name", "")
        reward = to_float(row.get("total_reward"))
        orders = to_int(row.get("orders_placed"))
        final_balance = to_float(row.get("final_balance"))

        stats = leaderboard.setdefault(name, {
            "strategy_name": name,
            "markets": 0,
            "total_reward": 0.0,
            "orders": 0,
            "wins": 0,
            "losses": 0,
            "best_reward": reward,
            "worst_reward": reward,
            "last_final_balance": final_balance,
        })
        stats["markets"] += 1
        stats["total_reward"] += reward
        stats["orders"] += orders
        stats["wins"] += 1 if reward > 0 else 0
        stats["losses"] += 1 if reward < 0 else 0
        stats["best_reward"] = max(stats["best_reward"], reward)
        stats["worst_reward"] = min(stats["worst_reward"], reward)
        stats["last_final_balance"] = final_balance

        market = market_rewards.setdefault(market_slug, {
            "market_slug": market_slug,
            "final_outcome": row.get("final_outcome", ""),
            "rewards": {},
        })
        market["rewards"][name] = reward

    leaders = sorted(leaderboard.values(), key=lambda row: row["total_reward"], reverse=True)
    for stats in leaders:
        markets = stats["markets"] or 1
        stats["mean_reward"] = stats["total_reward"] / markets
        stats["win_rate"] = stats["wins"] / markets
        stats["loss_rate"] = stats["losses"] / markets

    strategy_order = [row["strategy_name"] for row in leaders]
    cumulative = {name: 0.0 for name in strategy_order}
    paths = {name: [{"market_index": 0, "market_slug": "start", "cumulative_reward": 0.0}] for name in strategy_order}
    market_table = []
    for idx, market_slug in enumerate(market_order, start=1):
        market = market_rewards.get(market_slug, {"market_slug": market_slug, "final_outcome": "", "rewards": {}})
        rewards = {}
        for name in strategy_order:
            reward = to_float(market["rewards"].get(name), 0.0)
            rewards[name] = reward
            cumulative[name] += reward
            paths[name].append({
                "market_index": idx,
                "market_slug": market_slug,
                "cumulative_reward": cumulative[name],
            })
        market_table.append({
            "market_index": idx,
            "market_slug": market_slug,
            "final_outcome": market.get("final_outcome", ""),
            "rewards": rewards,
        })

    return {
        "leaderboard": leaders,
        "strategy_order": strategy_order,
        "market_rewards": market_table,
        "cumulative_paths": paths,
        "recent_rows": ok_rows[-100:][::-1],
    }


def cumulative_reward_svg(paths, strategy_order, width=1220, height=360, max_strategies=12):
    selected = strategy_order[:max_strategies]
    points = [point for name in selected for point in paths.get(name, [])]
    if not selected or not points:
        return "<p class=\"muted\">No completed market rewards yet.</p>"

    x_min = 0
    x_max = max(point["market_index"] for point in points) or 1
    y_values = [point["cumulative_reward"] for point in points] + [0.0]
    y_min = min(y_values)
    y_max = max(y_values)
    if y_min == y_max:
        y_min -= 1
        y_max += 1

    left, right, top, bottom = 70, 220, 34, 42
    chart_w = width - left - right
    chart_h = height - top - bottom
    colors = ["#8ec1ff", "#6fd38b", "#ffb86b", "#ff7d7d", "#c792ea", "#66d9ef", "#f78fb3", "#d7ba7d", "#9cdcfe", "#b5cea8", "#dcdcaa", "#ce9178"]

    def sx(x):
        return left + (x - x_min) / max(1, x_max - x_min) * chart_w

    def sy(y):
        return top + chart_h - (y - y_min) / (y_max - y_min) * chart_h

    parts = [
        f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"Cumulative reward paths\">",
        "<text x=\"16\" y=\"22\" class=\"chart-title\">Cumulative reward by strategy</text>",
        f"<line x1=\"{left}\" y1=\"{sy(0):.1f}\" x2=\"{left + chart_w}\" y2=\"{sy(0):.1f}\" class=\"zero-line\" />",
        f"<line x1=\"{left}\" y1=\"{top}\" x2=\"{left}\" y2=\"{top + chart_h}\" class=\"grid-line\" />",
        f"<line x1=\"{left}\" y1=\"{top + chart_h}\" x2=\"{left + chart_w}\" y2=\"{top + chart_h}\" class=\"grid-line\" />",
        f"<text x=\"8\" y=\"{sy(y_max):.1f}\" class=\"axis-label\">{y_max:.1f}</text>",
        f"<text x=\"8\" y=\"{sy(y_min):.1f}\" class=\"axis-label\">{y_min:.1f}</text>",
        f"<text x=\"{left}\" y=\"{height - 10}\" class=\"axis-label\">market 0</text>",
        f"<text x=\"{left + chart_w - 58}\" y=\"{height - 10}\" class=\"axis-label\">market {x_max}</text>",
    ]
    for i, name in enumerate(selected):
        series = paths.get(name, [])
        if len(series) < 2:
            continue
        point_text = " ".join(f"{sx(p['market_index']):.1f},{sy(p['cumulative_reward']):.1f}" for p in series)
        color = colors[i % len(colors)]
        parts.append(f"<polyline points=\"{point_text}\" fill=\"none\" stroke=\"{color}\" stroke-width=\"2\" />")
        end = series[-1]
        parts.append(f"<circle cx=\"{sx(end['market_index']):.1f}\" cy=\"{sy(end['cumulative_reward']):.1f}\" r=\"3\" fill=\"{color}\" />")
        legend_y = top + 18 + i * 20
        parts.append(f"<line x1=\"{left + chart_w + 18}\" y1=\"{legend_y}\" x2=\"{left + chart_w + 34}\" y2=\"{legend_y}\" stroke=\"{color}\" stroke-width=\"2\" />")
        parts.append(f"<text x=\"{left + chart_w + 40}\" y=\"{legend_y + 4}\" class=\"legend-label\">{html_escape(name)[:34]}</text>")
    parts.append("</svg>")
    return "".join(parts)


def html_escape(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_static_summary(run_dir, summary_path, summary):
    def esc(value):
        return html_escape(value)

    leaders = summary["leaderboard"]
    recent = summary["recent_rows"]
    market_rewards = summary["market_rewards"]
    strategy_order = summary["strategy_order"]
    leader_rows = "\n".join(
        f"<tr><td>{esc(row['strategy_name'])}</td><td>{row['markets']}</td><td>{row['mean_reward']:.4f}</td><td>{row['total_reward']:.4f}</td><td>{row['win_rate']:.1%}</td><td>{row['orders']}</td></tr>"
        for row in leaders
    )
    recent_rows = "\n".join(
        f"<tr><td>{esc(row.get('market_slug', ''))}</td><td>{esc(row.get('strategy_name', ''))}</td><td>{esc(row.get('final_outcome', ''))}</td><td>{esc(row.get('total_reward', ''))}</td><td>{esc(row.get('orders_placed', ''))}</td></tr>"
        for row in recent
    )
    reward_header = "".join(f"<th>{esc(name)}</th>" for name in strategy_order)
    reward_rows = "\n".join(
        "<tr>"
        f"<td>{row['market_index']}</td><td>{esc(row['market_slug'])}</td><td>{esc(row['final_outcome'])}</td>"
        + "".join(
            f"<td class=\"{'pos' if row['rewards'].get(name, 0) > 0 else 'neg' if row['rewards'].get(name, 0) < 0 else ''}\">{row['rewards'].get(name, 0):.4f}</td>"
            for name in strategy_order
        )
        + "</tr>"
        for row in market_rewards[-80:]
    )
    reward_path = cumulative_reward_svg(summary["cumulative_paths"], strategy_order)
    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Live Strategy Suite Summary</title>
  <style>
    body {{ margin:0; font-family: Arial, sans-serif; background:#0f1218; color:#e9edf5; }}
    header {{ padding:18px 22px; background:#171d29; border-bottom:1px solid #2a3345; }}
    main {{ padding:18px 22px; display:grid; grid-template-columns:1fr; gap:22px; }}
    .muted {{ color:#9aa5b7; font-size:13px; }}
    .chart-scroll, .table-scroll {{ overflow-x:auto; }}
    .chart-title {{ font:700 15px Arial, sans-serif; fill:#e9edf5; }}
    .axis-label, .legend-label {{ font:11px Arial, sans-serif; fill:#9aa5b7; }}
    .grid-line {{ stroke:#2a3345; stroke-width:1; }}
    .zero-line {{ stroke:#6a7485; stroke-width:1; stroke-dasharray:4 4; }}
    table {{ width:100%; border-collapse:collapse; background:#121824; }}
    th, td {{ padding:9px 10px; border-bottom:1px solid #2a3345; text-align:left; font-size:13px; }}
    th {{ color:#8ec1ff; }}
    .pos {{ color:#6fd38b; }}
    .neg {{ color:#ff7d7d; }}
  </style>
</head>
<body>
  <header>
    <h1>Live Strategy Suite Summary</h1>
    <div class="muted">Run: {esc(run_dir)}</div>
    <div class="muted">Summary CSV: {esc(summary_path)}</div>
  </header>
  <main>
    <section>
      <h2>Performance Summary</h2>
      <table><thead><tr><th>Strategy</th><th>Markets</th><th>Mean Reward</th><th>Total Reward</th><th>Win Rate</th><th>Orders</th></tr></thead><tbody>{leader_rows}</tbody></table>
    </section>
    <section>
      <h2>Balance Paths By Strategy</h2>
      <p class="muted">Each line adds the market reward in sequence, so the slope shows how the live paper account would compound across completed markets.</p>
      <div class="chart-scroll">{reward_path}</div>
    </section>
    <section>
      <h2>Reward By Market</h2>
      <div class="table-scroll"><table><thead><tr><th>#</th><th>Market</th><th>Outcome</th>{reward_header}</tr></thead><tbody>{reward_rows}</tbody></table></div>
    </section>
    <section>
      <h2>Recent Results</h2>
      <table><thead><tr><th>Market</th><th>Strategy</th><th>Outcome</th><th>Reward</th><th>Orders</th></tr></thead><tbody>{recent_rows}</tbody></table>
    </section>
  </main>
</body>
</html>"""


app = Flask(__name__) if Flask is not None else None
if app is not None:
    app.config["SECRET_KEY"] = "dev"
socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading") if SocketIO is not None else None
SUITE = None

DASHBOARD_HTML = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Live Strategy Suite</title>
  <script src="https://cdn.socket.io/4.7.5/socket.io.min.js"></script>
  <style>
    body { margin:0; font-family: Consolas, monospace; background:#0b0e13; color:#e8ecf3; }
    .top { display:flex; justify-content:space-between; gap:20px; padding:14px 18px; background:#131923; border-bottom:1px solid #273246; }
    .title { font-size:18px; font-weight:700; color:#8ec1ff; }
    .sub { color:#9aa6b8; font-size:12px; }
    .grid { display:grid; grid-template-columns: repeat(5, 1fr); gap:1px; background:#273246; }
    .cell { background:#101722; padding:14px 16px; min-height:64px; }
    .label { color:#8d98aa; font-size:11px; text-transform:uppercase; }
    .value { font-size:18px; margin-top:6px; overflow-wrap:anywhere; }
    .wide { grid-column: span 2; }
    .tables { display:grid; grid-template-columns: 1.2fr .8fr; gap:16px; padding:16px; }
    .below { padding:0 16px 20px; display:grid; gap:16px; }
    .chart-panel, .table-panel { background:#101722; padding:14px; overflow-x:auto; }
    table { width:100%; border-collapse:collapse; background:#101722; }
    th, td { padding:8px 10px; border-bottom:1px solid #273246; font-size:12px; text-align:left; }
    th { color:#8ec1ff; }
    .green { color:#6fd38b; }
    .red { color:#ff7d7d; }
    .muted { color:#98a4b5; }
    .chart-title { font:700 15px Consolas, monospace; fill:#e8ecf3; }
    .axis-label, .legend-label { font:11px Consolas, monospace; fill:#98a4b5; }
    .grid-line { stroke:#273246; stroke-width:1; }
    .zero-line { stroke:#687386; stroke-width:1; stroke-dasharray:4 4; }
  </style>
</head>
<body>
  <div class="top">
    <div>
      <div class="title">Live Strategy Suite</div>
      <div class="sub" id="run"></div>
    </div>
    <div class="sub" id="summary"></div>
  </div>
  <div class="grid">
    <div class="cell wide"><div class="label">Market</div><div class="value" id="market">--</div></div>
    <div class="cell"><div class="label">Strategies</div><div class="value" id="count">--</div></div>
    <div class="cell"><div class="label">Markets Seen</div><div class="value" id="seen">--</div></div>
    <div class="cell"><div class="label">Time Left</div><div class="value" id="left">--</div></div>
    <div class="cell"><div class="label">Prices</div><div class="value" id="prices">--</div></div>
    <div class="cell"><div class="label">BTC</div><div class="value" id="btc">--</div></div>
    <div class="cell"><div class="label">Price To Beat</div><div class="value" id="ptb">--</div></div>
    <div class="cell"><div class="label">BTC Distance</div><div class="value" id="dist">--</div></div>
    <div class="cell wide"><div class="label">Current Slug</div><div class="value" id="slug">--</div></div>
  </div>
  <div class="tables">
    <div>
      <h3>Live Leaderboard</h3>
      <table>
        <thead><tr><th>Strategy</th><th>Reward</th><th>Market Bal</th><th>Master</th><th>Eff Start</th><th>Cash</th><th>UP</th><th>DOWN</th><th>Orders</th><th>Last</th></tr></thead>
        <tbody id="leaderboard"></tbody>
      </table>
    </div>
    <div>
      <h3>Recent Actions</h3>
      <table>
        <thead><tr><th>Time</th><th>Strategy</th><th>Action</th><th>Reason</th></tr></thead>
        <tbody id="actions"></tbody>
      </table>
    </div>
    <div>
      <h3>Completed Markets</h3>
      <table>
        <thead><tr><th>Market</th><th>Outcome</th><th>Top Strategy</th><th>Top Reward</th></tr></thead>
        <tbody id="completed"></tbody>
      </table>
    </div>
  </div>
  <div class="below">
    <div class="table-panel">
      <h3>Performance Summary</h3>
      <table>
        <thead><tr><th>Strategy</th><th>Markets</th><th>Mean Reward</th><th>Total Reward</th><th>Win Rate</th><th>Orders</th></tr></thead>
        <tbody id="completedStats"></tbody>
      </table>
    </div>
    <div class="chart-panel">
      <h3>Balance Paths By Strategy</h3>
      <div class="muted">Cumulative reward across completed live markets.</div>
      <div id="completedChart"></div>
    </div>
    <div class="table-panel">
      <h3>Reward By Market</h3>
      <div id="marketRewardTable"></div>
    </div>
  </div>
<script>
const socket = io();
function fmtMoney(x){ return x == null || x === '' ? '--' : Number(x).toFixed(2); }
function fmtNum(x, d=3){ return x == null || x === '' ? '--' : Number(x).toFixed(d); }
function clsReward(x){ return Number(x) > 0 ? 'green' : Number(x) < 0 ? 'red' : ''; }
function cumulativeSvg(summary){
  const order = (summary.strategy_order || []).slice(0, 12);
  const paths = summary.cumulative_paths || {};
  const all = order.flatMap(name => paths[name] || []);
  if (!order.length || !all.length) return '<div class="muted">No completed market rewards yet.</div>';
  const width = 1220, height = 360, left = 70, right = 220, top = 34, bottom = 42;
  const chartW = width - left - right, chartH = height - top - bottom;
  const xMax = Math.max(...all.map(p => Number(p.market_index) || 0), 1);
  const yVals = all.map(p => Number(p.cumulative_reward) || 0).concat([0]);
  let yMin = Math.min(...yVals), yMax = Math.max(...yVals);
  if (yMin === yMax) { yMin -= 1; yMax += 1; }
  const sx = x => left + (x / xMax) * chartW;
  const sy = y => top + chartH - ((y - yMin) / (yMax - yMin)) * chartH;
  const colors = ['#8ec1ff','#6fd38b','#ffb86b','#ff7d7d','#c792ea','#66d9ef','#f78fb3','#d7ba7d','#9cdcfe','#b5cea8','#dcdcaa','#ce9178'];
  let out = `<svg viewBox="0 0 ${width} ${height}" role="img"><text x="16" y="22" class="chart-title">Cumulative reward by strategy</text>`;
  out += `<line x1="${left}" y1="${sy(0).toFixed(1)}" x2="${left + chartW}" y2="${sy(0).toFixed(1)}" class="zero-line" />`;
  out += `<line x1="${left}" y1="${top}" x2="${left}" y2="${top + chartH}" class="grid-line" />`;
  out += `<line x1="${left}" y1="${top + chartH}" x2="${left + chartW}" y2="${top + chartH}" class="grid-line" />`;
  out += `<text x="8" y="${sy(yMax).toFixed(1)}" class="axis-label">${yMax.toFixed(1)}</text><text x="8" y="${sy(yMin).toFixed(1)}" class="axis-label">${yMin.toFixed(1)}</text>`;
  order.forEach((name, i) => {
    const series = paths[name] || [];
    if (series.length < 2) return;
    const color = colors[i % colors.length];
    const pts = series.map(p => `${sx(Number(p.market_index)||0).toFixed(1)},${sy(Number(p.cumulative_reward)||0).toFixed(1)}`).join(' ');
    out += `<polyline points="${pts}" fill="none" stroke="${color}" stroke-width="2" />`;
    const y = top + 18 + i * 20;
    out += `<line x1="${left + chartW + 18}" y1="${y}" x2="${left + chartW + 34}" y2="${y}" stroke="${color}" stroke-width="2" />`;
    out += `<text x="${left + chartW + 40}" y="${y + 4}" class="legend-label">${name.slice(0,34)}</text>`;
  });
  return out + '</svg>';
}
function renderCompletedSummary(summary){
  summary = summary || {};
  const leaders = summary.leaderboard || [];
  document.getElementById('completedStats').innerHTML = leaders.map(r =>
    `<tr><td>${r.strategy_name}</td><td>${r.markets}</td><td>${fmtMoney(r.mean_reward)}</td><td>${fmtMoney(r.total_reward)}</td><td>${fmtNum((r.win_rate || 0) * 100,1)}%</td><td>${r.orders}</td></tr>`
  ).join('');
  document.getElementById('completedChart').innerHTML = cumulativeSvg(summary);
  const order = summary.strategy_order || [];
  const header = '<table><thead><tr><th>#</th><th>Market</th><th>Outcome</th>' + order.map(s => `<th>${s}</th>`).join('') + '</tr></thead><tbody>';
  const rows = (summary.market_rewards || []).slice(-80).map(m =>
    '<tr><td>'+m.market_index+'</td><td>'+m.market_slug+'</td><td>'+m.final_outcome+'</td>' +
    order.map(s => `<td class="${clsReward(m.rewards[s])}">${fmtMoney(m.rewards[s] || 0)}</td>`).join('') + '</tr>'
  ).join('');
  document.getElementById('marketRewardTable').innerHTML = header + rows + '</tbody></table>';
}
socket.on('tick', d => {
  document.getElementById('run').textContent = d.run_dir || '--';
  document.getElementById('summary').textContent = d.summary_html || '--';
  document.getElementById('market').textContent = d.market_question || '--';
  document.getElementById('count').textContent = d.strategy_count ?? '--';
  document.getElementById('seen').textContent = d.markets_seen ?? '--';
  document.getElementById('left').textContent = d.seconds_left == null ? '--' : Number(d.seconds_left).toFixed(1) + 's';
  document.getElementById('prices').innerHTML = '<span class="green">' + fmtNum(d.up_price) + '</span> / <span class="red">' + fmtNum(d.down_price) + '</span>';
  document.getElementById('btc').textContent = fmtMoney(d.btc_price);
  document.getElementById('ptb').textContent = fmtMoney(d.price_to_beat) + (d.price_to_beat_source ? ' ' + d.price_to_beat_source : '');
  document.getElementById('dist').textContent = d.distance == null ? '--' : fmtMoney(d.distance);
  document.getElementById('slug').textContent = d.current_slug || '--';
  document.getElementById('leaderboard').innerHTML = (d.leaderboard || []).slice(0, 25).map(r =>
    '<tr><td>'+r.strategy_name+'</td><td>'+fmtMoney(r.reward)+'</td><td>'+fmtMoney(r.balance)+'</td><td>'+fmtMoney(r.master_balance)+'</td><td>'+fmtMoney(r.market_starting_balance)+'</td><td>'+fmtMoney(r.cash)+'</td><td>'+fmtNum(r.up_tokens,2)+'</td><td>'+fmtNum(r.down_tokens,2)+'</td><td>'+r.orders+'</td><td>'+r.last_action+'</td></tr>'
  ).join('');
  document.getElementById('actions').innerHTML = (d.latest_actions || []).slice(0, 16).map(a =>
    '<tr><td>'+a.timestamp+'</td><td>'+a.strategy_name+'</td><td>'+a.action+'</td><td>'+a.reason+'</td></tr>'
  ).join('');
  document.getElementById('completed').innerHTML = (d.recent_completed || []).map(m =>
    '<tr><td>'+m.market_slug+'</td><td>'+m.final_outcome+'</td><td>'+m.top_strategy+'</td><td>'+fmtMoney(m.top_reward)+'</td></tr>'
  ).join('');
  renderCompletedSummary(d.completed_summary);
});
</script>
</body>
</html>
"""


if app is not None:
    @app.route("/")
    def index():
        return render_template_string(DASHBOARD_HTML)


    @app.route("/summary")
    def summary():
        if SUITE and SUITE.summary_html_path.exists():
            return SUITE.summary_html_path.read_text(encoding="utf-8")
        return "No completed market summaries yet.", 200


def emit_tick_loop():
    if socketio is None:
        return
    while SUITE and SUITE.running:
        socketio.emit("tick", SUITE.dashboard_payload())
        time.sleep(0.5)


def main():
    global SUITE
    args = parse_args()
    if Flask is None or SocketIO is None:
        raise SystemExit("Missing live dashboard dependencies. Run: pip install -r requirements.txt")
    suite = LiveStrategySuite(args)
    SUITE = suite

    print("Live strategy suite paper trader")
    print("=" * 50)
    print(f"Strategy folder: {args.strategy_folder}")
    print(f"Strategies:      {len(suite.strategies)}")
    if args.balance_config:
        print(f"Balance config:  {args.balance_config}")
        print(f"Start balance:   {suite.base_starting_balance:.2f} master balance per strategy")
    else:
        print(f"Start balance:   {args.starting_balance:.2f} per strategy per market")
    print(f"Target fallback: first {args.max_target_fallback_elapsed:.1f}s only")
    print(f"Dashboard:       http://localhost:{args.port}")
    print(f"Summary page:    http://localhost:{args.port}/summary")
    print(f"Run dir:         {suite.run_dir}")
    print(f"Summary CSV:     {suite.summary_path}")

    threading.Thread(target=start_binance_loop, args=(suite,), daemon=True).start()
    threading.Thread(target=start_chainlink_loop, args=(suite,), daemon=True).start()
    threading.Thread(target=poll_clob_loop, args=(suite,), daemon=True).start()
    threading.Thread(target=emit_tick_loop, daemon=True).start()

    try:
        socketio.run(app, host="0.0.0.0", port=args.port, debug=False, allow_unsafe_werkzeug=True)
    except KeyboardInterrupt:
        print("Stopping...")
    finally:
        suite.running = False
        suite.end_market()


if __name__ == "__main__":
    main()
