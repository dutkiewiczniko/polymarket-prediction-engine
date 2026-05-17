import argparse
import html
import json
import math
from pathlib import Path

import pandas as pd


DEFAULT_INPUT_DIR = Path("market_data")
DEFAULT_OUTPUT_DIR = Path("analysis/market_end_swings")
CSV_PATTERN = "btc-updown-5m-*.csv"
DEFAULT_FINAL_WINDOWS = [60.0, 30.0, 15.0, 10.0, 5.0, 2.0, 1.0]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze late BTC-vs-price_to_beat direction swings in replayable market CSVs."
    )
    parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help=f"Folder containing market CSVs. Default: {DEFAULT_INPUT_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Folder for analysis outputs. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--late-window-seconds",
        type=float,
        default=15.0,
        help="Window used for the main 'late swing' classification. Default: 15",
    )
    return parser.parse_args()


def parse_float_series(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def infer_btc_price(df: pd.DataFrame) -> pd.Series:
    btc_chainlink = parse_float_series(df.get("btc_chainlink", pd.Series(index=df.index, dtype=float)))
    btc_binance = parse_float_series(df.get("btc_binance", pd.Series(index=df.index, dtype=float)))
    return btc_chainlink.where(btc_chainlink.notna(), btc_binance)


def infer_signal(df: pd.DataFrame) -> pd.Series:
    btc_price = infer_btc_price(df)
    price_to_beat = parse_float_series(df.get("price_to_beat", pd.Series(index=df.index, dtype=float)))
    signal = pd.Series(index=df.index, dtype=object)
    valid = btc_price.notna() & price_to_beat.notna()
    signal.loc[valid] = btc_price.loc[valid].ge(price_to_beat.loc[valid]).map({True: "UP", False: "DOWN"})
    return signal


def value_at_or_before(df: pd.DataFrame, elapsed_value: float, column: str):
    rows = df.loc[df["elapsed"] <= elapsed_value, column].dropna()
    if rows.empty:
        return None
    return rows.iloc[-1]


def summarize_distribution(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "p10": None,
            "p25": None,
            "p75": None,
            "p90": None,
            "max": None,
        }
    series = pd.Series(values, dtype=float)
    return {
        "count": int(series.count()),
        "mean": float(series.mean()),
        "median": float(series.median()),
        "min": float(series.min()),
        "p10": float(series.quantile(0.10)),
        "p25": float(series.quantile(0.25)),
        "p75": float(series.quantile(0.75)),
        "p90": float(series.quantile(0.90)),
        "max": float(series.max()),
    }


def detect_outliers(values: list[float]) -> tuple[float | None, float | None]:
    if len(values) < 4:
        return None, None
    series = pd.Series(values, dtype=float)
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    return q1 - 1.5 * iqr, q3 + 1.5 * iqr


def analyze_market(csv_path: Path, late_window_seconds: float, final_windows: list[float]) -> dict[str, object] | None:
    df = pd.read_csv(csv_path)
    required = {"elapsed", "seconds_left", "price_to_beat"}
    if not required.issubset(df.columns):
        return None

    df = df.copy()
    df["elapsed"] = parse_float_series(df["elapsed"])
    df["seconds_left"] = parse_float_series(df["seconds_left"])
    df["up_price"] = parse_float_series(df.get("up_price", pd.Series(index=df.index, dtype=float)))
    df["down_price"] = parse_float_series(df.get("down_price", pd.Series(index=df.index, dtype=float)))
    df["price_to_beat"] = parse_float_series(df.get("price_to_beat", pd.Series(index=df.index, dtype=float)))
    df["btc_price"] = infer_btc_price(df)
    df["signal"] = infer_signal(df)
    df = df.dropna(subset=["elapsed"]).sort_values(["elapsed"]).reset_index(drop=True)
    valid = df.dropna(subset=["signal", "btc_price", "price_to_beat"])
    if valid.empty:
        return None

    valid = valid.loc[:, ["elapsed", "seconds_left", "up_price", "down_price", "btc_price", "price_to_beat", "signal"]].copy()
    end_rows = valid.loc[valid["seconds_left"].notna() & valid["seconds_left"].le(0)]
    if not end_rows.empty:
        market_end_elapsed = float(end_rows["elapsed"].iloc[0])
        valid = valid.loc[valid["elapsed"] <= market_end_elapsed].copy()
    else:
        market_end_elapsed = float(valid["elapsed"].max())
    valid["signal_changed"] = valid["signal"].ne(valid["signal"].shift(1))
    change_rows = valid.loc[valid["signal_changed"] & valid["signal"].shift(1).notna()].copy()

    final_elapsed = market_end_elapsed
    final_seconds_left = float(valid["seconds_left"].dropna().iloc[-1]) if valid["seconds_left"].dropna().any() else None
    final_signal = str(valid["signal"].iloc[-1])
    final_btc = float(valid["btc_price"].iloc[-1])
    final_price_to_beat = float(valid["price_to_beat"].iloc[-1])
    final_up_price = float(valid["up_price"].iloc[-1]) if pd.notna(valid["up_price"].iloc[-1]) else None
    final_down_price = float(valid["down_price"].iloc[-1]) if pd.notna(valid["down_price"].iloc[-1]) else None
    final_btc_distance = final_btc - final_price_to_beat
    final_btc_distance_pct = (final_btc_distance / final_price_to_beat) if final_price_to_beat else None

    if change_rows.empty:
        last_swing_elapsed = None
        seconds_before_end_last_swing = None
        last_swing_from = None
        last_swing_to = None
        last_swing_up_price = None
        last_swing_down_price = None
        last_swing_btc_price = None
        last_swing_price_to_beat = None
        last_swing_btc_distance = None
    else:
        last_change = change_rows.iloc[-1]
        last_index = int(last_change.name)
        last_swing_elapsed = float(last_change["elapsed"])
        seconds_before_end_last_swing = final_elapsed - last_swing_elapsed
        last_swing_from = str(valid.loc[last_index - 1, "signal"]) if last_index > 0 else None
        last_swing_to = str(last_change["signal"])
        last_swing_up_price = float(last_change["up_price"]) if pd.notna(last_change["up_price"]) else None
        last_swing_down_price = float(last_change["down_price"]) if pd.notna(last_change["down_price"]) else None
        last_swing_btc_price = float(last_change["btc_price"]) if pd.notna(last_change["btc_price"]) else None
        last_swing_price_to_beat = float(last_change["price_to_beat"]) if pd.notna(last_change["price_to_beat"]) else None
        last_swing_btc_distance = (
            last_swing_btc_price - last_swing_price_to_beat
            if last_swing_btc_price is not None and last_swing_price_to_beat not in {None, 0}
            else None
        )

    signals_by_window: dict[str, str | None] = {}
    changed_vs_final: dict[str, bool | None] = {}
    for window in final_windows:
        checkpoint = final_elapsed - window
        signal_value = value_at_or_before(valid, checkpoint, "signal")
        key = f"{int(window) if float(window).is_integer() else window:g}s"
        signals_by_window[key] = None if signal_value is None or pd.isna(signal_value) else str(signal_value)
        changed_vs_final[key] = None if signal_value is None or pd.isna(signal_value) else str(signal_value) != final_signal

    return {
        "market_id": csv_path.stem,
        "file_path": str(csv_path),
        "rows": int(len(df)),
        "valid_signal_rows": int(len(valid)),
        "final_elapsed": final_elapsed,
        "final_seconds_left": final_seconds_left,
        "final_signal": final_signal,
        "final_up_price": final_up_price,
        "final_down_price": final_down_price,
        "final_btc_price": final_btc,
        "final_price_to_beat": final_price_to_beat,
        "final_btc_distance_to_beat": final_btc_distance,
        "final_btc_distance_to_beat_pct": final_btc_distance_pct,
        "swing_count": int(len(change_rows)),
        "had_any_swing": bool(len(change_rows) > 0),
        "had_late_swing": bool(seconds_before_end_last_swing is not None and seconds_before_end_last_swing <= late_window_seconds),
        "last_swing_elapsed": last_swing_elapsed,
        "seconds_before_end_last_swing": seconds_before_end_last_swing,
        "last_swing_from": last_swing_from,
        "last_swing_to": last_swing_to,
        "last_swing_up_price": last_swing_up_price,
        "last_swing_down_price": last_swing_down_price,
        "last_swing_btc_price": last_swing_btc_price,
        "last_swing_price_to_beat": last_swing_price_to_beat,
        "last_swing_btc_distance_to_beat": last_swing_btc_distance,
        **{f"signal_at_{key}": value for key, value in signals_by_window.items()},
        **{f"changed_vs_final_from_{key}": value for key, value in changed_vs_final.items()},
    }


def build_summary(per_market_df: pd.DataFrame, late_window_seconds: float, final_windows: list[float]) -> dict[str, object]:
    total_markets = int(len(per_market_df))
    any_swing_df = per_market_df[per_market_df["had_any_swing"] == True].copy()
    no_swing_df = per_market_df[per_market_df["had_any_swing"] != True].copy()
    late_swing_df = per_market_df[per_market_df["had_late_swing"] == True].copy()

    last_swing_values = [
        float(value)
        for value in any_swing_df["seconds_before_end_last_swing"].dropna().tolist()
    ]
    lower_outlier, upper_outlier = detect_outliers(last_swing_values)

    summary = {
        "total_markets": total_markets,
        "markets_with_any_swing": int(len(any_swing_df)),
        "markets_without_any_swing": int(len(no_swing_df)),
        "markets_with_late_swing": int(len(late_swing_df)),
        "late_window_seconds": late_window_seconds,
        "percent_with_any_swing": float(len(any_swing_df) / total_markets) if total_markets else 0.0,
        "percent_without_any_swing": float(len(no_swing_df) / total_markets) if total_markets else 0.0,
        "percent_with_late_swing": float(len(late_swing_df) / total_markets) if total_markets else 0.0,
        "last_swing_seconds_before_end_distribution": summarize_distribution(last_swing_values),
        "last_swing_outlier_bounds_seconds_before_end": {
            "lower": lower_outlier,
            "upper": upper_outlier,
        },
        "latest_swings": any_swing_df.sort_values("seconds_before_end_last_swing", ascending=True).head(10)[
            ["market_id", "seconds_before_end_last_swing", "last_swing_from", "last_swing_to", "swing_count"]
        ].to_dict("records"),
        "earliest_last_swings": any_swing_df.sort_values("seconds_before_end_last_swing", ascending=False).head(10)[
            ["market_id", "seconds_before_end_last_swing", "last_swing_from", "last_swing_to", "swing_count"]
        ].to_dict("records"),
        "high_swing_count_markets": per_market_df.sort_values("swing_count", ascending=False).head(10)[
            ["market_id", "swing_count", "seconds_before_end_last_swing", "final_signal"]
        ].to_dict("records"),
    }

    window_stats = {}
    for window in final_windows:
        key = f"{int(window) if float(window).is_integer() else window:g}s"
        column = f"changed_vs_final_from_{key}"
        known = per_market_df[column].dropna()
        changed = int((known == True).sum())
        same = int((known == False).sum())
        window_stats[key] = {
            "rows_with_comparison": int(len(known)),
            "changed_vs_final_count": changed,
            "same_as_final_count": same,
            "changed_vs_final_pct": float(changed / len(known)) if len(known) else 0.0,
        }
    summary["window_change_stats"] = window_stats

    if lower_outlier is not None and upper_outlier is not None:
        summary["outlier_markets"] = any_swing_df[
            (any_swing_df["seconds_before_end_last_swing"] < lower_outlier)
            | (any_swing_df["seconds_before_end_last_swing"] > upper_outlier)
        ][["market_id", "seconds_before_end_last_swing", "swing_count", "last_swing_from", "last_swing_to"]].to_dict("records")
    else:
        summary["outlier_markets"] = []

    return summary


def print_summary(summary: dict[str, object]) -> None:
    dist = summary["last_swing_seconds_before_end_distribution"]
    print(f"Markets processed:         {summary['total_markets']}")
    print(f"Markets with any swing:    {summary['markets_with_any_swing']}")
    print(f"Markets without swing:     {summary['markets_without_any_swing']}")
    print(f"Markets with late swing:   {summary['markets_with_late_swing']} (<= {summary['late_window_seconds']:.1f}s before end)")
    print()
    print("Last swing seconds before end:")
    print(f"  mean   {fmt(dist['mean'])}")
    print(f"  median {fmt(dist['median'])}")
    print(f"  min    {fmt(dist['min'])}")
    print(f"  p10    {fmt(dist['p10'])}")
    print(f"  p25    {fmt(dist['p25'])}")
    print(f"  p75    {fmt(dist['p75'])}")
    print(f"  p90    {fmt(dist['p90'])}")
    print(f"  max    {fmt(dist['max'])}")
    print()
    print("Changed vs final direction at checkpoints:")
    for key, stats in summary["window_change_stats"].items():
        print(
            f"  {key:>4}: changed {stats['changed_vs_final_count']:>3} / {stats['rows_with_comparison']:<3}"
            f" ({stats['changed_vs_final_pct'] * 100:5.1f}%)"
        )
    print()
    print(f"Outlier markets:           {len(summary['outlier_markets'])}")


def pct(value):
    if value is None:
        return "-"
    if isinstance(value, float) and not math.isfinite(value):
        return "-"
    return f"{float(value) * 100:.2f}%"


def table_html(df: pd.DataFrame, columns: list[str], formatters: dict[str, callable] | None = None, table_id: str = "table") -> str:
    if df.empty:
        return '<p class="muted">No rows.</p>'
    formatters = formatters or {}
    cols = [column for column in columns if column in df.columns]
    header = "".join(f'<th data-sortable="1">{html.escape(column)}<span class="sort-indicator"></span></th>' for column in cols)
    rows = []
    for _, row in df.loc[:, cols].iterrows():
        cells = []
        for column in cols:
            value = row[column]
            if column in formatters:
                value = formatters[column](value)
            elif pd.isna(value):
                value = ""
            cells.append(f"<td>{html.escape(str(value))}</td>")
        rows.append(f"<tr>{''.join(cells)}</tr>")
    return f'<table id="{html.escape(table_id)}" class="sortable"><thead><tr>{header}</tr></thead><tbody>{"".join(rows)}</tbody></table>'


def render_html_report(summary: dict[str, object], per_market_df: pd.DataFrame, skipped_df: pd.DataFrame, output_path: Path) -> None:
    late_df = per_market_df[per_market_df["had_late_swing"] == True].copy()
    no_swing_df = per_market_df[per_market_df["had_any_swing"] != True].copy()
    price_formatters = {
        "seconds_before_end_last_swing": fmt,
        "final_up_price": fmt,
        "final_down_price": fmt,
        "final_btc_price": fmt,
        "final_price_to_beat": fmt,
        "final_btc_distance_to_beat": fmt,
        "final_btc_distance_to_beat_pct": pct,
        "last_swing_up_price": fmt,
        "last_swing_down_price": fmt,
        "last_swing_btc_price": fmt,
        "last_swing_price_to_beat": fmt,
        "last_swing_btc_distance_to_beat": fmt,
    }
    late_columns = [
        "market_id",
        "seconds_before_end_last_swing",
        "swing_count",
        "last_swing_from",
        "last_swing_to",
        "last_swing_up_price",
        "last_swing_down_price",
        "last_swing_btc_price",
        "last_swing_price_to_beat",
        "last_swing_btc_distance_to_beat",
        "final_signal",
        "final_up_price",
        "final_down_price",
        "final_btc_price",
        "final_price_to_beat",
        "final_btc_distance_to_beat",
        "final_btc_distance_to_beat_pct",
    ]
    no_swing_columns = [
        "market_id",
        "final_signal",
        "final_up_price",
        "final_down_price",
        "final_btc_price",
        "final_price_to_beat",
        "final_btc_distance_to_beat",
        "final_btc_distance_to_beat_pct",
    ]
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Market End Swings Report</title>
  <style>
    body {{ margin: 0; font-family: Arial, sans-serif; background: #f5f7fb; color: #182230; }}
    header {{ padding: 28px 34px 18px; background: #111827; color: white; }}
    main {{ padding: 24px 34px 40px; }}
    section {{ background: white; border: 1px solid #d8dee9; border-radius: 8px; padding: 18px; margin-bottom: 18px; }}
    h1, h2, h3 {{ margin-top: 0; }}
    .cards {{ display: grid; grid-template-columns: repeat(5, minmax(140px, 1fr)); gap: 12px; }}
    .card {{ background: white; border: 1px solid #d8dee9; border-radius: 8px; padding: 14px; }}
    .label {{ color: #667085; font-size: 12px; text-transform: uppercase; }}
    .value {{ font-size: 22px; font-weight: 700; margin-top: 6px; }}
    .muted {{ color: #667085; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #d8dee9; padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #f8fafc; position: sticky; top: 0; }}
    .table-wrap {{ overflow-x: auto; }}
    table.sortable th {{ cursor: pointer; user-select: none; white-space: nowrap; }}
    .sort-indicator {{ color: #667085; font-size: 10px; margin-left: 5px; }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 16px; }}
    ul {{ margin: 0; padding-left: 18px; }}
  </style>
</head>
<body>
  <header>
    <h1>Market End Swings Report</h1>
    <div class="muted">Input: {html.escape(str(summary["input_dir"]))}</div>
  </header>
  <main>
    <div class="cards">
      <div class="card"><div class="label">Markets</div><div class="value">{summary["total_markets"]}</div></div>
      <div class="card"><div class="label">Any swing</div><div class="value">{summary["markets_with_any_swing"]}</div></div>
      <div class="card"><div class="label">No swing</div><div class="value">{summary["markets_without_any_swing"]}</div></div>
      <div class="card"><div class="label">Late swing</div><div class="value">{summary["markets_with_late_swing"]}</div></div>
      <div class="card"><div class="label">Skipped</div><div class="value">{summary["skipped_markets"]}</div></div>
    </div>

    <section>
      <h2>Summary</h2>
      <div class="grid-2">
        <div>
          <h3>Last swing timing</h3>
          <ul>
            <li>Mean: {fmt(summary["last_swing_seconds_before_end_distribution"]["mean"])}s before end</li>
            <li>Median: {fmt(summary["last_swing_seconds_before_end_distribution"]["median"])}s before end</li>
            <li>P10: {fmt(summary["last_swing_seconds_before_end_distribution"]["p10"])}s</li>
            <li>P25: {fmt(summary["last_swing_seconds_before_end_distribution"]["p25"])}s</li>
            <li>P75: {fmt(summary["last_swing_seconds_before_end_distribution"]["p75"])}s</li>
            <li>P90: {fmt(summary["last_swing_seconds_before_end_distribution"]["p90"])}s</li>
          </ul>
        </div>
        <div>
          <h3>Changed vs final direction</h3>
          <ul>
            {''.join(f'<li>{key}: {stats["changed_vs_final_count"]} / {stats["rows_with_comparison"]} changed ({stats["changed_vs_final_pct"] * 100:.1f}%)</li>' for key, stats in summary["window_change_stats"].items())}
          </ul>
        </div>
      </div>
    </section>

    <section>
      <h2>Latest Late Swings</h2>
      <p class="muted">These are the markets that changed direction closest to the end, with final prices and BTC-vs-threshold context.</p>
      <div class="table-wrap">{table_html(late_df.sort_values("seconds_before_end_last_swing").head(30), late_columns, price_formatters, "late-swings")}</div>
    </section>

    <section>
      <h2>No Swing Markets</h2>
      <p class="muted">Markets that never changed inferred BTC-vs-threshold direction during the recorded replayable window.</p>
      <div class="table-wrap">{table_html(no_swing_df, no_swing_columns, price_formatters, "no-swings")}</div>
    </section>

    <section>
      <h2>Outlier Markets</h2>
      <div class="table-wrap">{table_html(pd.DataFrame(summary["outlier_markets"]), ["market_id", "seconds_before_end_last_swing", "swing_count", "last_swing_from", "last_swing_to"], {"seconds_before_end_last_swing": fmt}, "outliers")}</div>
    </section>

    <section>
      <h2>Skipped Files</h2>
      <div class="table-wrap">{table_html(skipped_df, ["market_id", "file_path", "reason"], table_id="skipped")}</div>
    </section>
  </main>
  <script>
    function parseCellValue(text) {{
      const cleaned = text.trim().replace(/,/g, "");
      if (cleaned.endsWith("%")) {{
        const pctValue = Number(cleaned.slice(0, -1));
        return Number.isNaN(pctValue) ? cleaned.toLowerCase() : pctValue;
      }}
      const numeric = Number(cleaned);
      return cleaned !== "" && !Number.isNaN(numeric) ? numeric : cleaned.toLowerCase();
    }}
    function compareValues(a, b) {{
      if (typeof a === "number" && typeof b === "number") return a - b;
      return String(a).localeCompare(String(b), undefined, {{ numeric: true, sensitivity: "base" }});
    }}
    document.querySelectorAll("table.sortable").forEach(table => {{
      const headers = table.querySelectorAll("th[data-sortable='1']");
      headers.forEach((header, columnIndex) => {{
        header.addEventListener("click", () => {{
          const currentDirection = header.dataset.sortDirection === "asc" ? "desc" : "asc";
          headers.forEach(item => {{
            item.dataset.sortDirection = "";
            const indicator = item.querySelector(".sort-indicator");
            if (indicator) indicator.textContent = "";
          }});
          header.dataset.sortDirection = currentDirection;
          const indicator = header.querySelector(".sort-indicator");
          if (indicator) indicator.textContent = currentDirection === "asc" ? "▲" : "▼";
          const tbody = table.querySelector("tbody");
          const rows = Array.from(tbody.querySelectorAll("tr"));
          rows.sort((rowA, rowB) => {{
            const a = parseCellValue(rowA.children[columnIndex]?.textContent || "");
            const b = parseCellValue(rowB.children[columnIndex]?.textContent || "");
            const result = compareValues(a, b);
            return currentDirection === "asc" ? result : -result;
          }});
          rows.forEach(row => tbody.appendChild(row));
        }});
      }});
    }});
  </script>
</body>
</html>
"""
    output_path.write_text(html_text, encoding="utf-8")


def fmt(value):
    if value is None:
        return "-"
    if isinstance(value, float) and not math.isfinite(value):
        return "-"
    return f"{float(value):.3f}"


def fallback_locked_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 1000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"Could not find fallback path for {path}")


def main():
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_paths = sorted(input_dir.glob(CSV_PATTERN))
    if not csv_paths:
        raise FileNotFoundError(f"No market files found in {input_dir} matching {CSV_PATTERN}")

    per_market_rows = []
    skipped = []
    for csv_path in csv_paths:
        try:
            row = analyze_market(csv_path, args.late_window_seconds, DEFAULT_FINAL_WINDOWS)
        except Exception as exc:
            row = None
            skipped.append({"market_id": csv_path.stem, "file_path": str(csv_path), "reason": f"{type(exc).__name__}: {exc}"})
        if row is None:
            skipped.append({"market_id": csv_path.stem, "file_path": str(csv_path), "reason": "missing required usable columns/rows"})
            continue
        per_market_rows.append(row)

    per_market_df = pd.DataFrame(per_market_rows).sort_values(
        ["had_late_swing", "seconds_before_end_last_swing", "swing_count"],
        ascending=[False, True, False],
        na_position="last",
    )
    summary = build_summary(per_market_df, args.late_window_seconds, DEFAULT_FINAL_WINDOWS)
    summary["input_dir"] = str(input_dir)
    summary["skipped_markets"] = len(skipped)

    per_market_path = output_dir / "market_end_swings_per_market.csv"
    skipped_path = output_dir / "market_end_swings_skipped.csv"
    summary_path = output_dir / "market_end_swings_summary.json"
    html_path = output_dir / "market_end_swings_report.html"

    skipped_df = pd.DataFrame(skipped)
    try:
        per_market_df.to_csv(per_market_path, index=False)
    except PermissionError:
        per_market_path = fallback_locked_path(per_market_path)
        per_market_df.to_csv(per_market_path, index=False)
    try:
        skipped_df.to_csv(skipped_path, index=False)
    except PermissionError:
        skipped_path = fallback_locked_path(skipped_path)
        skipped_df.to_csv(skipped_path, index=False)
    try:
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except PermissionError:
        summary_path = fallback_locked_path(summary_path)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    try:
        render_html_report(summary, per_market_df, skipped_df, html_path)
    except PermissionError:
        html_path = fallback_locked_path(html_path)
        render_html_report(summary, per_market_df, skipped_df, html_path)

    print_summary(summary)
    print()
    print(f"Per-market CSV: {per_market_path}")
    print(f"Skipped CSV:    {skipped_path}")
    print(f"Summary JSON:   {summary_path}")
    print(f"HTML report:    {html_path}")


if __name__ == "__main__":
    main()
