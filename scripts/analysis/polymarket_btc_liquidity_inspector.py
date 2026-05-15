import argparse
import csv
import hashlib
import html
import json
import math
import statistics
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import requests


GAMMA_API = "https://gamma-api.polymarket.com"
CLOB_API = "https://clob.polymarket.com"
DEFAULT_OUTPUT_ROOT = Path("runs/liquidity_inspection")
DEPTH_WINDOWS_CENTS = (1, 2, 5)


SNAPSHOT_FIELDS = [
    "timestamp_utc",
    "unix_time",
    "market_slug",
    "question",
    "seconds_left",
    "market_start",
    "market_end",
    "price_to_beat",
    "gamma_volume",
    "gamma_liquidity",
    "side",
    "token_id",
    "price_buy",
    "price_sell",
    "best_bid",
    "best_ask",
    "midpoint",
    "spread",
    "book_hash",
    "bid_levels",
    "ask_levels",
    "bid_size_total",
    "ask_size_total",
    "bid_usd_total",
    "ask_usd_total",
    "ask_size_within_1c",
    "ask_usd_within_1c",
    "bid_size_within_1c",
    "bid_usd_within_1c",
    "ask_size_within_2c",
    "ask_usd_within_2c",
    "bid_size_within_2c",
    "bid_usd_within_2c",
    "ask_size_within_5c",
    "ask_usd_within_5c",
    "bid_size_within_5c",
    "bid_usd_within_5c",
    "changed_from_previous_side_snapshot",
]


LEVEL_FIELDS = [
    "timestamp_utc",
    "unix_time",
    "market_slug",
    "side",
    "token_id",
    "book_side",
    "level_index",
    "price",
    "size",
    "usd_notional",
    "distance_from_best",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect read-only Polymarket BTC 5-minute orderbook snapshots and "
            "summarize realistic visible liquidity for simulator sizing."
        )
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT), help="Root folder for collected data.")
    parser.add_argument("--run-id", help="Optional run folder name. Defaults to timestamped id.")
    parser.add_argument("--interval-seconds", type=float, default=2.5, help="Polling interval. Default: 2.5.")
    parser.add_argument("--duration-minutes", type=float, default=30.0, help="How long to collect. Default: 30.")
    parser.add_argument("--max-snapshots", type=int, default=0, help="Stop after this many market snapshots. 0 means duration only.")
    parser.add_argument("--request-timeout", type=float, default=5.0, help="HTTP timeout in seconds.")
    parser.add_argument("--start-immediately", action="store_true", help="Do not wait for market start if discovered early.")
    parser.add_argument(
        "--continue-on-errors",
        action="store_true",
        help="Keep polling through transient HTTP or parsing errors.",
    )
    parser.add_argument(
        "--report-only",
        help="Generate report for an existing run folder without collecting new snapshots.",
    )
    return parser.parse_args()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        if math.isfinite(float(value)):
            return float(value)
        return None
    text = str(value).strip().replace(",", "")
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def safe_int_time(value: Any) -> float | None:
    number = safe_float(value)
    if number is not None:
        return number
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def extract_first_number(payload: Any, keys: tuple[str, ...]) -> float | None:
    if isinstance(payload, dict):
        for key in keys:
            value = safe_float(payload.get(key))
            if value is not None:
                return value
        for value in payload.values():
            found = extract_first_number(value, keys)
            if found is not None:
                return found
    elif isinstance(payload, list):
        for item in payload:
            found = extract_first_number(item, keys)
            if found is not None:
                return found
    return None


def parse_token_ids(raw: Any) -> list[str]:
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except json.JSONDecodeError:
            return []
    if not isinstance(raw, list):
        return []
    return [str(item) for item in raw if item is not None]


def market_slug_for_slot(slot: int) -> str:
    return f"btc-updown-5m-{slot}"


def discover_active_btc_market(session: requests.Session, timeout: float, slot: int | None = None) -> dict | None:
    now = int(time.time())
    base_slot = slot if slot is not None else now - (now % 300)
    for offset in (0, 300, -300, 600):
        candidate_slot = base_slot + offset
        slug = market_slug_for_slot(candidate_slot)
        response = session.get(f"{GAMMA_API}/events", params={"slug": slug}, timeout=timeout)
        response.raise_for_status()
        events = response.json()
        if not events:
            continue
        event = events[0]
        markets = event.get("markets") or []
        if not markets:
            continue
        market = markets[0]
        token_ids = parse_token_ids(market.get("clobTokenIds"))
        if len(token_ids) < 2:
            continue
        start_time = safe_int_time(event.get("startTime") or market.get("startDate"))
        end_time = safe_int_time(market.get("endDate") or event.get("endTime"))
        if end_time is None or end_time <= now:
            continue

        return {
            "slug": slug,
            "slot": candidate_slot,
            "question": market.get("question") or event.get("title") or slug,
            "up_token": token_ids[0],
            "down_token": token_ids[1],
            "start_time": start_time or float(candidate_slot),
            "end_time": end_time,
            "price_to_beat": extract_first_number(
                {"event": event, "market": market},
                (
                    "priceToBeat",
                    "price_to_beat",
                    "strike",
                    "strikePrice",
                    "targetPrice",
                    "initialPrice",
                ),
            ),
            "gamma_volume": extract_first_number(market, ("volume", "volumeNum", "volumeClob", "volume24hr")),
            "gamma_liquidity": extract_first_number(market, ("liquidity", "liquidityNum")),
            "raw_event": event,
            "raw_market": market,
        }
    return None


def fetch_json(session: requests.Session, url: str, timeout: float, *, params: dict | None = None, method: str = "GET", json_body=None):
    if method == "POST":
        response = session.post(url, params=params, json=json_body, timeout=timeout)
    else:
        response = session.get(url, params=params, timeout=timeout)
    response.raise_for_status()
    return response.json()


def fetch_price(session: requests.Session, token_id: str, side: str, timeout: float) -> float | None:
    try:
        payload = fetch_json(session, f"{CLOB_API}/price", timeout, params={"token_id": token_id, "side": side})
    except requests.RequestException:
        return None
    return safe_float(payload.get("price") if isinstance(payload, dict) else None)


def fetch_order_book(session: requests.Session, token_id: str, timeout: float) -> dict:
    try:
        return fetch_json(session, f"{CLOB_API}/book", timeout, params={"token_id": token_id})
    except requests.RequestException:
        # Older clients and some docs examples use token_id; keep asset_id fallback for compatibility.
        return fetch_json(session, f"{CLOB_API}/book", timeout, params={"asset_id": token_id})


def normalize_levels(levels: Any, reverse: bool) -> list[dict]:
    normalized = []
    if not isinstance(levels, list):
        return normalized
    for item in levels:
        if not isinstance(item, dict):
            continue
        price = safe_float(item.get("price"))
        size = safe_float(item.get("size"))
        if price is None or size is None or price <= 0 or size <= 0:
            continue
        normalized.append({"price": price, "size": size, "usd_notional": price * size})
    return sorted(normalized, key=lambda level: level["price"], reverse=reverse)


def depth_within(levels: list[dict], best_price: float | None, cents: int, *, ask_side: bool) -> tuple[float, float]:
    if best_price is None:
        return 0.0, 0.0
    window = cents / 100.0
    if ask_side:
        selected = [level for level in levels if level["price"] <= best_price + window + 1e-12]
    else:
        selected = [level for level in levels if level["price"] >= best_price - window - 1e-12]
    return (
        sum(level["size"] for level in selected),
        sum(level["usd_notional"] for level in selected),
    )


def summarize_book(book: dict) -> dict:
    bids = normalize_levels(book.get("bids"), reverse=True)
    asks = normalize_levels(book.get("asks"), reverse=False)
    best_bid = bids[0]["price"] if bids else None
    best_ask = asks[0]["price"] if asks else None
    midpoint = (best_bid + best_ask) / 2 if best_bid is not None and best_ask is not None else None
    spread = best_ask - best_bid if best_bid is not None and best_ask is not None else None
    result = {
        "bids": bids,
        "asks": asks,
        "best_bid": best_bid,
        "best_ask": best_ask,
        "midpoint": midpoint,
        "spread": spread,
        "book_hash": str(book.get("hash") or ""),
        "bid_levels": len(bids),
        "ask_levels": len(asks),
        "bid_size_total": sum(level["size"] for level in bids),
        "ask_size_total": sum(level["size"] for level in asks),
        "bid_usd_total": sum(level["usd_notional"] for level in bids),
        "ask_usd_total": sum(level["usd_notional"] for level in asks),
    }
    if not result["book_hash"]:
        encoded = json.dumps({"bids": bids, "asks": asks}, sort_keys=True, separators=(",", ":")).encode("utf-8")
        result["book_hash"] = hashlib.sha256(encoded).hexdigest()
    for cents in DEPTH_WINDOWS_CENTS:
        ask_size, ask_usd = depth_within(asks, best_ask, cents, ask_side=True)
        bid_size, bid_usd = depth_within(bids, best_bid, cents, ask_side=False)
        result[f"ask_size_within_{cents}c"] = ask_size
        result[f"ask_usd_within_{cents}c"] = ask_usd
        result[f"bid_size_within_{cents}c"] = bid_size
        result[f"bid_usd_within_{cents}c"] = bid_usd
    return result


@dataclass
class OutputFiles:
    run_dir: Path
    snapshots_csv: Path
    levels_csv: Path
    raw_jsonl: Path
    report_html: Path


def create_output_files(output_root: str | Path, run_id: str | None) -> OutputFiles:
    output_root = Path(output_root)
    run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_btc_liquidity")
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return OutputFiles(
        run_dir=run_dir,
        snapshots_csv=run_dir / "snapshot_metrics.csv",
        levels_csv=run_dir / "orderbook_levels.csv",
        raw_jsonl=run_dir / "raw_snapshots.jsonl",
        report_html=run_dir / "liquidity_report.html",
    )


def open_csv_writer(path: Path, fields: list[str]):
    exists = path.exists() and path.stat().st_size > 0
    handle = path.open("a", newline="", encoding="utf-8")
    writer = csv.DictWriter(handle, fieldnames=fields)
    if not exists:
        writer.writeheader()
        handle.flush()
    return handle, writer


def flatten_level_rows(snapshot_row: dict, book_summary: dict, book_side: str) -> list[dict]:
    levels = book_summary["asks"] if book_side == "ask" else book_summary["bids"]
    best = book_summary["best_ask"] if book_side == "ask" else book_summary["best_bid"]
    rows = []
    for idx, level in enumerate(levels, start=1):
        rows.append({
            "timestamp_utc": snapshot_row["timestamp_utc"],
            "unix_time": snapshot_row["unix_time"],
            "market_slug": snapshot_row["market_slug"],
            "side": snapshot_row["side"],
            "token_id": snapshot_row["token_id"],
            "book_side": book_side,
            "level_index": idx,
            "price": level["price"],
            "size": level["size"],
            "usd_notional": level["usd_notional"],
            "distance_from_best": abs(level["price"] - best) if best is not None else "",
        })
    return rows


def build_snapshot_rows(session: requests.Session, market: dict, timeout: float, previous_hashes: dict[str, str]) -> tuple[list[dict], list[dict], dict]:
    timestamp = utc_now()
    unix_time = time.time()
    seconds_left = market["end_time"] - unix_time
    rows = []
    level_rows = []
    raw = {
        "timestamp_utc": timestamp,
        "unix_time": unix_time,
        "market": {key: value for key, value in market.items() if key not in {"raw_event", "raw_market"}},
        "tokens": {},
    }

    for side, token_id in (("up", market["up_token"]), ("down", market["down_token"])):
        price_buy = fetch_price(session, token_id, "buy", timeout)
        price_sell = fetch_price(session, token_id, "sell", timeout)
        book = fetch_order_book(session, token_id, timeout)
        summary = summarize_book(book)
        previous_hash = previous_hashes.get(token_id)
        changed = "" if previous_hash is None else previous_hash != summary["book_hash"]
        previous_hashes[token_id] = summary["book_hash"]

        row = {
            "timestamp_utc": timestamp,
            "unix_time": unix_time,
            "market_slug": market["slug"],
            "question": market["question"],
            "seconds_left": seconds_left,
            "market_start": market["start_time"],
            "market_end": market["end_time"],
            "price_to_beat": market.get("price_to_beat"),
            "gamma_volume": market.get("gamma_volume"),
            "gamma_liquidity": market.get("gamma_liquidity"),
            "side": side,
            "token_id": token_id,
            "price_buy": price_buy,
            "price_sell": price_sell,
            "best_bid": summary["best_bid"],
            "best_ask": summary["best_ask"],
            "midpoint": summary["midpoint"],
            "spread": summary["spread"],
            "book_hash": summary["book_hash"],
            "bid_levels": summary["bid_levels"],
            "ask_levels": summary["ask_levels"],
            "bid_size_total": summary["bid_size_total"],
            "ask_size_total": summary["ask_size_total"],
            "bid_usd_total": summary["bid_usd_total"],
            "ask_usd_total": summary["ask_usd_total"],
            "changed_from_previous_side_snapshot": changed,
        }
        for cents in DEPTH_WINDOWS_CENTS:
            row[f"ask_size_within_{cents}c"] = summary[f"ask_size_within_{cents}c"]
            row[f"ask_usd_within_{cents}c"] = summary[f"ask_usd_within_{cents}c"]
            row[f"bid_size_within_{cents}c"] = summary[f"bid_size_within_{cents}c"]
            row[f"bid_usd_within_{cents}c"] = summary[f"bid_usd_within_{cents}c"]
        rows.append(row)
        level_rows.extend(flatten_level_rows(row, summary, "bid"))
        level_rows.extend(flatten_level_rows(row, summary, "ask"))
        raw["tokens"][side] = {
            "token_id": token_id,
            "price_buy": price_buy,
            "price_sell": price_sell,
            "book": book,
            "summary": {key: value for key, value in summary.items() if key not in {"bids", "asks"}},
        }

    return rows, level_rows, raw


def collect_snapshots(args) -> Path:
    outputs = create_output_files(args.output_root, args.run_id)
    session = requests.Session()
    snapshot_handle, snapshot_writer = open_csv_writer(outputs.snapshots_csv, SNAPSHOT_FIELDS)
    level_handle, level_writer = open_csv_writer(outputs.levels_csv, LEVEL_FIELDS)
    raw_handle = outputs.raw_jsonl.open("a", encoding="utf-8")
    deadline = time.time() + args.duration_minutes * 60.0
    max_snapshots = int(args.max_snapshots or 0)
    snapshots_written = 0
    current_market = None
    previous_hashes: dict[str, str] = {}

    print(f"Output folder: {outputs.run_dir}")
    print(f"Polling interval: {args.interval_seconds:.2f}s")
    print("Mode: read-only public market data collection")

    try:
        while time.time() < deadline:
            try:
                if current_market is None or time.time() >= current_market["end_time"]:
                    current_market = discover_active_btc_market(session, args.request_timeout)
                    previous_hashes = {}
                    if current_market is None:
                        print("No active BTC 5-minute market found; retrying.")
                        time.sleep(args.interval_seconds)
                        continue
                    print(
                        f"Market: {current_market['slug']} | "
                        f"{current_market['question']} | "
                        f"ends in {current_market['end_time'] - time.time():.1f}s"
                    )
                    if not args.start_immediately:
                        wait_s = current_market["start_time"] - time.time()
                        if wait_s > 0:
                            time.sleep(min(wait_s, args.interval_seconds))
                            continue

                rows, level_rows, raw = build_snapshot_rows(
                    session,
                    current_market,
                    args.request_timeout,
                    previous_hashes,
                )
                for row in rows:
                    snapshot_writer.writerow(row)
                for row in level_rows:
                    level_writer.writerow(row)
                raw_handle.write(json.dumps(raw, separators=(",", ":"), default=str) + "\n")
                snapshot_handle.flush()
                level_handle.flush()
                raw_handle.flush()
                snapshots_written += 1
                up_row = next((row for row in rows if row["side"] == "up"), rows[0])
                print(
                    f"{snapshots_written:05d} {current_market['slug']} "
                    f"up_bid={fmt_num(up_row['best_bid'], 3)} "
                    f"up_ask={fmt_num(up_row['best_ask'], 3)} "
                    f"spread={fmt_num(up_row['spread'], 3)} "
                    f"ask_2c_usd={fmt_num(up_row['ask_usd_within_2c'], 0)}"
                )
                if max_snapshots and snapshots_written >= max_snapshots:
                    break
            except Exception as exc:
                print(f"Collection error: {type(exc).__name__}: {exc}")
                if not args.continue_on_errors:
                    raise
            time.sleep(args.interval_seconds)
    finally:
        snapshot_handle.close()
        level_handle.close()
        raw_handle.close()

    generate_report(outputs.run_dir)
    return outputs.run_dir


def load_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def numeric_values(rows: list[dict], column: str) -> list[float]:
    values = []
    for row in rows:
        value = safe_float(row.get(column))
        if value is not None:
            values.append(value)
    return values


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    pos = (len(ordered) - 1) * q
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)


def mean(values: list[float]) -> float | None:
    return statistics.fmean(values) if values else None


def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def fmt_num(value: Any, digits: int = 2) -> str:
    value = safe_float(value)
    if value is None:
        return ""
    if abs(value) >= 1000:
        return f"{value:,.{digits}f}"
    return f"{value:.{digits}f}"


def fmt_pct(value: Any) -> str:
    value = safe_float(value)
    if value is None:
        return ""
    return f"{value * 100:.1f}%"


def group_rows(rows: list[dict], key: str) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(str(row.get(key, "")), []).append(row)
    return grouped


def table_html(headers: list[str], rows: list[list[Any]], numeric_indexes: set[int] | None = None) -> str:
    numeric_indexes = numeric_indexes or set()
    if not rows:
        return '<p class="muted">No rows.</p>'
    head = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    body = []
    for row in rows:
        cells = []
        for idx, value in enumerate(row):
            cls = ' class="num"' if idx in numeric_indexes else ""
            cells.append(f"<td{cls}>{html.escape(str(value))}</td>")
        body.append(f"<tr>{''.join(cells)}</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(body)}</tbody></table>"


def build_overall_rows(rows: list[dict]) -> list[list[str]]:
    grouped = group_rows(rows, "side")
    output = []
    for side in ("up", "down"):
        side_rows = grouped.get(side, [])
        if not side_rows:
            continue
        output.append([
            side,
            len(side_rows),
            fmt_num(mean(numeric_values(side_rows, "spread")), 4),
            fmt_num(median(numeric_values(side_rows, "spread")), 4),
            fmt_num(mean(numeric_values(side_rows, "ask_usd_within_1c")), 0),
            fmt_num(median(numeric_values(side_rows, "ask_usd_within_1c")), 0),
            fmt_num(mean(numeric_values(side_rows, "ask_usd_within_2c")), 0),
            fmt_num(median(numeric_values(side_rows, "ask_usd_within_2c")), 0),
            fmt_num(mean(numeric_values(side_rows, "ask_usd_within_5c")), 0),
            fmt_num(median(numeric_values(side_rows, "ask_usd_within_5c")), 0),
        ])
    return output


def build_market_rows(rows: list[dict]) -> list[list[str]]:
    output = []
    for slug, market_rows in group_rows(rows, "market_slug").items():
        output.append([
            slug,
            len({row["timestamp_utc"] for row in market_rows}),
            fmt_num(mean(numeric_values(market_rows, "gamma_volume")), 0),
            fmt_num(mean(numeric_values(market_rows, "gamma_liquidity")), 0),
            fmt_num(mean(numeric_values(market_rows, "spread")), 4),
            fmt_num(mean(numeric_values(market_rows, "ask_usd_within_1c")), 0),
            fmt_num(mean(numeric_values(market_rows, "ask_usd_within_2c")), 0),
            fmt_num(mean(numeric_values(market_rows, "ask_usd_within_5c")), 0),
        ])
    return sorted(output, key=lambda row: safe_float(row[6].replace(",", "")) or 0.0, reverse=True)


def build_change_rows(rows: list[dict]) -> list[list[str]]:
    output = []
    for token_key, token_rows in group_rows(rows, "token_id").items():
        if not token_key:
            continue
        side = token_rows[0].get("side", "")
        sorted_rows = sorted(token_rows, key=lambda row: safe_float(row.get("unix_time")) or 0.0)
        comparable = [row for row in sorted_rows if str(row.get("changed_from_previous_side_snapshot")) in {"True", "False"}]
        changes = [row for row in comparable if str(row.get("changed_from_previous_side_snapshot")) == "True"]
        times = [safe_float(row.get("unix_time")) for row in changes]
        times = [value for value in times if value is not None]
        intervals = [b - a for a, b in zip(times, times[1:]) if b >= a]
        output.append([
            side,
            token_key[:18] + "...",
            len(sorted_rows),
            len(changes),
            fmt_pct(len(changes) / len(comparable) if comparable else None),
            fmt_num(mean(intervals), 2),
        ])
    return output


def build_example_rows(rows: list[dict], column: str, reverse: bool) -> list[list[str]]:
    sorted_rows = sorted(
        rows,
        key=lambda row: safe_float(row.get(column)) if safe_float(row.get(column)) is not None else -1.0,
        reverse=reverse,
    )
    output = []
    seen = set()
    for row in sorted_rows:
        key = (row.get("market_slug"), row.get("side"), row.get("timestamp_utc"))
        if key in seen:
            continue
        seen.add(key)
        output.append([
            row.get("timestamp_utc", ""),
            row.get("market_slug", ""),
            row.get("side", ""),
            fmt_num(row.get("best_bid"), 3),
            fmt_num(row.get("best_ask"), 3),
            fmt_num(row.get("spread"), 4),
            fmt_num(row.get("ask_usd_within_1c"), 0),
            fmt_num(row.get("ask_usd_within_2c"), 0),
            fmt_num(row.get("ask_usd_within_5c"), 0),
        ])
        if len(output) >= 8:
            break
    return output


def suggested_order_ranges(rows: list[dict]) -> list[list[str]]:
    output = []
    for cents in DEPTH_WINDOWS_CENTS:
        values = numeric_values(rows, f"ask_usd_within_{cents}c")
        p10 = percentile(values, 0.10)
        p25 = percentile(values, 0.25)
        p50 = percentile(values, 0.50)
        output.append([
            f"within {cents}c of best ask",
            fmt_num(p10, 0),
            fmt_num(p25, 0),
            fmt_num(p50, 0),
            fmt_num((p10 or 0) * 0.25, 0),
            fmt_num((p25 or 0) * 0.25, 0),
            "Use smaller values for marketable/FOK orders; visible depth can vanish before fill.",
        ])
    return output


def generate_report(run_folder: str | Path) -> Path:
    run_folder = Path(run_folder)
    rows = load_csv(run_folder / "snapshot_metrics.csv")
    report_path = run_folder / "liquidity_report.html"
    ok_rows = [row for row in rows if row.get("best_bid") or row.get("best_ask")]
    market_count = len({row.get("market_slug") for row in ok_rows if row.get("market_slug")})
    snapshot_count = len({row.get("timestamp_utc") for row in ok_rows if row.get("timestamp_utc")})

    overall_headers = [
        "Side",
        "Rows",
        "Mean spread",
        "Median spread",
        "Mean ask USD 1c",
        "Median ask USD 1c",
        "Mean ask USD 2c",
        "Median ask USD 2c",
        "Mean ask USD 5c",
        "Median ask USD 5c",
    ]
    market_headers = [
        "Market",
        "Snapshots",
        "Gamma volume",
        "Gamma liquidity",
        "Mean spread",
        "Mean ask USD 1c",
        "Mean ask USD 2c",
        "Mean ask USD 5c",
    ]
    change_headers = ["Side", "Token", "Snapshots", "Book changes", "Change rate", "Mean seconds between changes"]
    example_headers = ["Time", "Market", "Side", "Best bid", "Best ask", "Spread", "Ask USD 1c", "Ask USD 2c", "Ask USD 5c"]
    suggestion_headers = ["Depth window", "P10 visible USD", "P25 visible USD", "Median visible USD", "Very conservative cap", "Conservative cap", "Note"]

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Polymarket BTC Liquidity Report</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; color: #152033; background: #f5f7fb; }}
    header {{ background: #101828; color: white; padding: 26px 34px; }}
    main {{ padding: 24px 34px 42px; }}
    section {{ background: white; border: 1px solid #d8dee9; border-radius: 8px; padding: 18px; margin-bottom: 18px; }}
    h1 {{ margin: 0 0 8px; font-size: 25px; }}
    h2 {{ margin: 0 0 12px; font-size: 18px; }}
    .muted {{ color: #667085; }}
    .cards {{ display: grid; grid-template-columns: repeat(4, minmax(140px, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .card {{ background: white; border: 1px solid #d8dee9; border-radius: 8px; padding: 14px; }}
    .label {{ color: #667085; font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .value {{ font-size: 22px; font-weight: 700; margin-top: 6px; }}
    .table-wrap {{ overflow-x: auto; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e4e7ec; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #f8fafc; white-space: nowrap; }}
    td.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
    code {{ background: #eef2f6; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <header>
    <h1>Polymarket BTC 5-Minute Liquidity Report</h1>
    <div class="muted">Run folder: {html.escape(str(run_folder))}</div>
  </header>
  <main>
    <div class="cards">
      <div class="card"><div class="label">Markets</div><div class="value">{market_count}</div></div>
      <div class="card"><div class="label">Snapshots</div><div class="value">{snapshot_count}</div></div>
      <div class="card"><div class="label">Side rows</div><div class="value">{len(ok_rows)}</div></div>
      <div class="card"><div class="label">Generated</div><div class="value">{html.escape(utc_now()[:19])}</div></div>
    </div>

    <section>
      <h2>Suggested Simulation Order Caps</h2>
      <p class="muted">These are based on visible ask-side USD depth. The suggested caps deliberately use a fraction of visible depth because public REST snapshots do not guarantee fills, queue priority, or stable liquidity.</p>
      <div class="table-wrap">{table_html(suggestion_headers, suggested_order_ranges(ok_rows), {1, 2, 3, 4, 5})}</div>
    </section>

    <section>
      <h2>Overall Depth And Spread</h2>
      <div class="table-wrap">{table_html(overall_headers, build_overall_rows(ok_rows), {1, 2, 3, 4, 5, 6, 7, 8, 9})}</div>
    </section>

    <section>
      <h2>Market-Level Averages</h2>
      <div class="table-wrap">{table_html(market_headers, build_market_rows(ok_rows), {1, 2, 3, 4, 5, 6, 7})}</div>
    </section>

    <section>
      <h2>Order Book Change Rate</h2>
      <p class="muted">Uses the orderbook hash from Polymarket when present, otherwise a hash of normalized bid/ask levels.</p>
      <div class="table-wrap">{table_html(change_headers, build_change_rows(ok_rows), {2, 3, 4, 5})}</div>
    </section>

    <section>
      <h2>Thin Market Examples</h2>
      <div class="table-wrap">{table_html(example_headers, build_example_rows(ok_rows, "ask_usd_within_2c", reverse=False), {3, 4, 5, 6, 7, 8})}</div>
    </section>

    <section>
      <h2>Liquid Market Examples</h2>
      <div class="table-wrap">{table_html(example_headers, build_example_rows(ok_rows, "ask_usd_within_2c", reverse=True), {3, 4, 5, 6, 7, 8})}</div>
    </section>

    <section>
      <h2>Files</h2>
      <p><code>snapshot_metrics.csv</code> contains one row per token side per snapshot.</p>
      <p><code>orderbook_levels.csv</code> contains each visible bid/ask price level with size and notional.</p>
      <p><code>raw_snapshots.jsonl</code> keeps raw CLOB/Gamma payloads for auditing parser assumptions.</p>
    </section>
  </main>
</body>
</html>
"""
    report_path.write_text(html_text, encoding="utf-8")
    print(f"Report: {report_path}")
    return report_path


def main():
    args = parse_args()
    if args.report_only:
        generate_report(args.report_only)
        return
    collect_snapshots(args)


if __name__ == "__main__":
    main()
