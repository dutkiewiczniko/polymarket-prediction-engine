import argparse
import html
import math
import random
import re
from pathlib import Path

import pandas as pd


DEFAULT_RUNS_DIR = Path("runs")
REPORT_FILENAME = "strategy_report.html"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Create an HTML strategy performance report from a batch run summary.csv."
    )
    parser.add_argument(
        "--run-folder",
        help="Batch run folder containing summary.csv. Defaults to newest runs/* folder with summary.csv.",
    )
    parser.add_argument(
        "--output",
        help="Output HTML path. Defaults to <run-folder>/strategy_report.html.",
    )
    parser.add_argument(
        "--examples-per-strategy",
        type=int,
        default=3,
        help="Best and worst summary rows to show for each strategy.",
    )
    parser.add_argument(
        "--max-trajectory-files-per-strategy",
        type=int,
        default=5,
        help="Trajectory files to sample per strategy for action stats and example rows.",
    )
    parser.add_argument(
        "--sample-plot-seed",
        type=int,
        default=42,
        help="Random seed for choosing one embedded plot per strategy family.",
    )
    parser.add_argument(
        "--max-sample-plots",
        type=int,
        default=50,
        help="Maximum embedded trajectory plots to include.",
    )
    parser.add_argument(
        "--no-sample-plots",
        action="store_true",
        help="Do not embed random trajectory plots in the HTML report.",
    )
    return parser.parse_args()


def newest_run_folder(runs_dir=DEFAULT_RUNS_DIR) -> Path:
    candidates = [
        path for path in Path(runs_dir).iterdir()
        if path.is_dir() and (path / "summary.csv").exists()
    ]
    if not candidates:
        raise FileNotFoundError("No runs/*/summary.csv files found.")
    return max(candidates, key=lambda path: (path / "summary.csv").stat().st_mtime)


def load_summary(run_folder: Path) -> pd.DataFrame:
    summary_path = run_folder / "summary.csv"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary.csv: {summary_path}")

    df = pd.read_csv(summary_path)
    if "status" not in df.columns:
        df["status"] = "ok"
    if "strategy_run_name" not in df.columns:
        df["strategy_run_name"] = df.get("strategy_name", "")
    if "strategy_name" not in df.columns:
        df["strategy_name"] = df["strategy_run_name"]
    if "output_csv" not in df.columns:
        df["output_csv"] = ""

    for column in ["final_balance", "starting_balance", "total_reward", "rows_written"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    df["strategy_key"] = df["strategy_run_name"].fillna(df["strategy_name"]).astype(str)
    return df


def build_strategy_stats(summary_df: pd.DataFrame) -> pd.DataFrame:
    ok_df = summary_df[summary_df["status"].fillna("ok").eq("ok")].copy()
    if ok_df.empty:
        return pd.DataFrame()

    grouped = ok_df.groupby("strategy_key", dropna=False)
    stats = grouped.agg(
        runs=("total_reward", "count"),
        markets=("market_file", "nunique") if "market_file" in ok_df.columns else ("total_reward", "count"),
        mean_reward=("total_reward", "mean"),
        median_reward=("total_reward", "median"),
        total_reward=("total_reward", "sum"),
        reward_std=("total_reward", "std"),
        min_reward=("total_reward", "min"),
        max_reward=("total_reward", "max"),
        mean_final_balance=("final_balance", "mean"),
        mean_rows=("rows_written", "mean"),
    ).reset_index()

    win_rate = grouped["total_reward"].apply(lambda s: float((s > 0).mean()))
    loss_rate = grouped["total_reward"].apply(lambda s: float((s < 0).mean()))
    flat_rate = grouped["total_reward"].apply(lambda s: float((s == 0).mean()))
    stats = stats.merge(win_rate.rename("win_rate"), on="strategy_key")
    stats = stats.merge(loss_rate.rename("loss_rate"), on="strategy_key")
    stats = stats.merge(flat_rate.rename("flat_rate"), on="strategy_key")
    stats["reward_std"] = stats["reward_std"].fillna(0.0)
    return stats.sort_values(["mean_reward", "win_rate"], ascending=[False, False])


def read_trajectory_sample(path: str | Path) -> pd.DataFrame | None:
    path = Path(path)
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except Exception:
        return None


def strategy_family(strategy_key: str) -> str:
    """Collapse generated variants into one family name for report plots."""
    value = str(strategy_key)
    value = re.sub(r"__\d+$", "", value)
    value = re.sub(r"_tp[^_]+_cheap[^_]+_(tok\d+|balpct\d+)$", "", value)
    return value


def extract_strategy_dimensions(strategy_key: str) -> dict[str, object]:
    value = str(strategy_key)
    family = strategy_family(value)
    match = re.search(r"_tp(?P<tp>[^_]+)_cheap(?P<cheap>[^_]+)_(?P<size>tok\d+|balpct\d+)$", value)
    if not match:
        return {
            "strategy_family": family,
            "tp_code": "",
            "cheap_code": "",
            "size_code": "",
            "size_mode": "",
        }
    size_code = match.group("size")
    return {
        "strategy_family": family,
        "tp_code": f"tp{match.group('tp')}",
        "cheap_code": f"cheap{match.group('cheap')}",
        "size_code": size_code,
        "size_mode": "token" if size_code.startswith("tok") else "balance_pct",
    }


def enrich_strategy_dimensions(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "strategy_key" not in df.columns:
        return df.copy()
    enriched = df.copy()
    dims = pd.DataFrame([extract_strategy_dimensions(value) for value in enriched["strategy_key"]])
    dims.index = enriched.index
    for column in dims.columns:
        enriched[column] = dims[column]
    return enriched


def build_action_stats(summary_df: pd.DataFrame, max_files_per_strategy: int) -> pd.DataFrame:
    rows = []
    ok_df = summary_df[summary_df["status"].fillna("ok").eq("ok")].copy()

    for strategy_key, group in ok_df.groupby("strategy_key", dropna=False):
        action_counts = {}
        event_rows = 0
        sampled_rows = 0
        sampled_files = 0
        example_reasons = []

        for output_csv in group["output_csv"].dropna().head(max_files_per_strategy):
            df = read_trajectory_sample(output_csv)
            if df is None or df.empty:
                continue
            sampled_files += 1
            sampled_rows += len(df)

            action_col = "action_executed" if "action_executed" in df.columns else "action"
            if action_col in df.columns:
                for action, count in df[action_col].fillna("").astype(str).value_counts().items():
                    action_counts[action] = action_counts.get(action, 0) + int(count)

            if "events_count" in df.columns:
                event_rows += int(pd.to_numeric(df["events_count"], errors="coerce").fillna(0).gt(0).sum())
            elif action_col in df.columns:
                event_rows += int(df[action_col].fillna("hold").astype(str).ne("hold").sum())

            if "reason" in df.columns:
                reasons = df.loc[df[action_col].fillna("hold").astype(str).ne("hold"), "reason"].dropna().astype(str)
                example_reasons.extend(reasons.head(3).tolist())

        if sampled_files:
            rows.append({
                "strategy_key": strategy_key,
                "sampled_files": sampled_files,
                "sampled_rows": sampled_rows,
                "hold_rows": action_counts.get("hold", 0),
                "buy_up_rows": action_counts.get("buy_up", 0),
                "buy_down_rows": action_counts.get("buy_down", 0),
                "other_action_rows": sum(
                    count for action, count in action_counts.items()
                    if action not in {"hold", "buy_up", "buy_down", ""}
                ),
                "event_rows": event_rows,
                "example_reasons": "; ".join(dict.fromkeys(example_reasons[:5])),
            })

    return pd.DataFrame(rows)


def build_dimension_stats(summary_df: pd.DataFrame, dimension: str) -> pd.DataFrame:
    ok_df = summary_df[summary_df["status"].fillna("ok").eq("ok")].copy()
    if ok_df.empty or dimension not in ok_df.columns:
        return pd.DataFrame()

    ok_df[dimension] = ok_df[dimension].fillna("").astype(str)
    ok_df = ok_df[ok_df[dimension] != ""]
    if ok_df.empty:
        return pd.DataFrame()

    grouped = ok_df.groupby(dimension, dropna=False)
    stats = grouped.agg(
        strategies=("strategy_key", "nunique"),
        runs=("total_reward", "count"),
        markets=("market_file", "nunique") if "market_file" in ok_df.columns else ("total_reward", "count"),
        mean_reward=("total_reward", "mean"),
        median_reward=("total_reward", "median"),
        total_reward=("total_reward", "sum"),
        reward_std=("total_reward", "std"),
        mean_final_balance=("final_balance", "mean"),
        min_reward=("total_reward", "min"),
        max_reward=("total_reward", "max"),
    ).reset_index()
    stats["win_rate"] = grouped["total_reward"].apply(lambda s: float((s > 0).mean())).values
    stats["loss_rate"] = grouped["total_reward"].apply(lambda s: float((s < 0).mean())).values
    stats["flat_rate"] = grouped["total_reward"].apply(lambda s: float((s == 0).mean())).values
    stats["reward_std"] = stats["reward_std"].fillna(0.0)
    return stats.sort_values(["mean_reward", "win_rate"], ascending=[False, False])


def build_family_size_pivot(summary_df: pd.DataFrame) -> pd.DataFrame:
    ok_df = summary_df[summary_df["status"].fillna("ok").eq("ok")].copy()
    required = {"strategy_family", "size_code", "total_reward", "final_balance"}
    if ok_df.empty or not required.issubset(ok_df.columns):
        return pd.DataFrame()

    ok_df = ok_df[
        ok_df["strategy_family"].fillna("").astype(str).ne("")
        & ok_df["size_code"].fillna("").astype(str).ne("")
    ].copy()
    if ok_df.empty:
        return pd.DataFrame()

    pivot = ok_df.pivot_table(
        index="strategy_family",
        columns="size_code",
        values=["total_reward", "final_balance"],
        aggfunc="mean",
    )
    if pivot.empty:
        return pd.DataFrame()
    pivot.columns = [f"{metric}_mean_{size}" for metric, size in pivot.columns]
    result = pivot.reset_index()
    reward_columns = [col for col in result.columns if col.startswith("total_reward_mean_")]
    if len(reward_columns) >= 2:
        reward_columns = sorted(reward_columns)
        result["reward_spread"] = result[reward_columns[-1]] - result[reward_columns[0]]
    return result


def pick_sample_plot_rows(summary_df: pd.DataFrame, seed: int, max_plots: int) -> list[dict]:
    ok_df = summary_df[summary_df["status"].fillna("ok").eq("ok")].copy()
    if ok_df.empty or "output_csv" not in ok_df.columns:
        return []

    ok_df = ok_df[ok_df["output_csv"].fillna("").astype(str).ne("")]
    ok_df["strategy_family"] = ok_df["strategy_key"].map(strategy_family)

    rng = random.Random(seed)
    samples = []
    for family, group in sorted(ok_df.groupby("strategy_family", dropna=False), key=lambda item: str(item[0])):
        rows = group.to_dict("records")
        rng.shuffle(rows)
        for row in rows:
            output_csv = Path(str(row.get("output_csv", "")))
            if output_csv.exists():
                row["strategy_family"] = family
                samples.append(row)
                break
        if len(samples) >= max_plots:
            break
    return samples


def pick_example_rows(summary_df: pd.DataFrame, examples_per_strategy: int) -> pd.DataFrame:
    ok_df = summary_df[summary_df["status"].fillna("ok").eq("ok")].copy()
    if ok_df.empty:
        return pd.DataFrame()

    examples = []
    for strategy_key, group in ok_df.groupby("strategy_key", dropna=False):
        ordered = group.sort_values("total_reward", ascending=False)
        examples.append(ordered.head(examples_per_strategy).assign(example_type="best"))
        examples.append(ordered.tail(examples_per_strategy).sort_values("total_reward").assign(example_type="worst"))
    return pd.concat(examples, ignore_index=True)


def pick_trajectory_examples(summary_df: pd.DataFrame, max_files_per_strategy: int) -> pd.DataFrame:
    rows = []
    ok_df = summary_df[summary_df["status"].fillna("ok").eq("ok")].copy()
    for strategy_key, group in ok_df.groupby("strategy_key", dropna=False):
        for output_csv in group["output_csv"].dropna().head(max_files_per_strategy):
            df = read_trajectory_sample(output_csv)
            if df is None or df.empty:
                continue
            action_col = "action_executed" if "action_executed" in df.columns else "action"
            if action_col not in df.columns:
                continue
            interesting = df[df[action_col].fillna("hold").astype(str).ne("hold")].head(3)
            if interesting.empty:
                interesting = df.head(1)
            for _, row in interesting.iterrows():
                rows.append({
                    "strategy_key": strategy_key,
                    "trajectory": Path(output_csv).name,
                    "elapsed": row.get("elapsed", ""),
                    "action": row.get(action_col, ""),
                    "reason": row.get("reason", ""),
                    "balance_before": row.get("balance_before", ""),
                    "balance_after": row.get("balance_after", ""),
                    "final_balance": row.get("final_balance", ""),
                    "total_reward": row.get("total_reward", ""),
                })
            break
    return pd.DataFrame(rows)


def fmt(value, digits=3):
    if value is None or value == "":
        return ""
    try:
        if pd.isna(value):
            return ""
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isfinite(number):
        return f"{number:.{digits}f}"
    return ""


def pct(value):
    try:
        if pd.isna(value):
            return ""
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return ""


def table_html(df: pd.DataFrame, columns: list[str], formatters=None, limit=None, table_id: str | None = None, sortable=False) -> str:
    if df.empty:
        return "<p class=\"muted\">No rows found.</p>"
    formatters = formatters or {}
    data = df.loc[:, [column for column in columns if column in df.columns]]
    if limit:
        data = data.head(limit)

    header_cells = []
    for column in data.columns:
        sort_attrs = " data-sortable=\"1\"" if sortable else ""
        header_cells.append(f"<th{sort_attrs}>{html.escape(column)}<span class=\"sort-indicator\"></span></th>")
    header = "".join(header_cells)
    body_rows = []
    for _, row in data.iterrows():
        cells = []
        for column in data.columns:
            value = row[column]
            if column in formatters:
                value = formatters[column](value)
            cells.append(f"<td>{html.escape(str(value))}</td>")
        body_rows.append(f"<tr>{''.join(cells)}</tr>")
    id_attr = f" id=\"{html.escape(table_id)}\"" if table_id else ""
    sortable_class = " sortable" if sortable else ""
    return f"<table{id_attr} class=\"{sortable_class.strip()}\"><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def chart_label_width(labels: pd.Series, minimum=190, maximum=430) -> int:
    longest = max((len(str(label)) for label in labels), default=0)
    return max(minimum, min(maximum, longest * 7 + 28))


def bar_chart_svg(stats: pd.DataFrame, value_column: str, title: str, width=1220, height=320) -> str:
    if stats.empty or value_column not in stats.columns:
        return "<p class=\"muted\">No chart data.</p>"

    data = stats[["strategy_key", value_column]].copy().head(20)
    data[value_column] = pd.to_numeric(data[value_column], errors="coerce").fillna(0.0)
    max_abs = max(float(data[value_column].abs().max()), 1.0)
    left = chart_label_width(data["strategy_key"])
    right = 24
    top = 32
    row_h = 22
    height = max(height, top + len(data) * row_h + 36)
    width = max(width, left + 620)
    chart_w = width - left - right
    zero_x = left + chart_w / 2

    parts = [
        f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"{html.escape(title)}\">",
        f"<text x=\"16\" y=\"22\" class=\"chart-title\">{html.escape(title)}</text>",
        f"<line x1=\"{zero_x:.1f}\" y1=\"30\" x2=\"{zero_x:.1f}\" y2=\"{height - 18}\" class=\"axis\" />",
    ]
    for idx, (_, row) in enumerate(data.iterrows()):
        y = top + idx * row_h
        label = str(row["strategy_key"])
        value = float(row[value_column])
        bar_w = abs(value) / max_abs * (chart_w / 2)
        if value >= 0:
            x = zero_x
            cls = "bar positive"
        else:
            x = zero_x - bar_w
            cls = "bar negative"
        parts.append(f"<text x=\"12\" y=\"{y + 14}\" class=\"axis-label\"><title>{html.escape(label)}</title>{html.escape(label)}</text>")
        parts.append(f"<rect x=\"{x:.1f}\" y=\"{y + 4}\" width=\"{bar_w:.1f}\" height=\"14\" class=\"{cls}\" />")
        parts.append(f"<text x=\"{x + bar_w + 5 if value >= 0 else x - 58:.1f}\" y=\"{y + 15}\" class=\"bar-value\">{fmt(value)}</text>")
    parts.append("</svg>")
    return "".join(parts)


def win_rate_svg(stats: pd.DataFrame, width=1220, height=320) -> str:
    if stats.empty:
        return "<p class=\"muted\">No chart data.</p>"

    data = stats[["strategy_key", "win_rate", "loss_rate", "flat_rate"]].copy().head(20)
    left = chart_label_width(data["strategy_key"])
    right = 24
    top = 32
    row_h = 22
    height = max(height, top + len(data) * row_h + 36)
    width = max(width, left + 620)
    chart_w = width - left - right
    parts = [
        f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"Win/loss rate\">",
        "<text x=\"16\" y=\"22\" class=\"chart-title\">Win / loss / flat rate</text>",
    ]
    for idx, (_, row) in enumerate(data.iterrows()):
        y = top + idx * row_h
        x = left
        label = str(row["strategy_key"])
        parts.append(f"<text x=\"12\" y=\"{y + 14}\" class=\"axis-label\"><title>{html.escape(label)}</title>{html.escape(label)}</text>")
        for column, cls in [("win_rate", "win"), ("flat_rate", "flat"), ("loss_rate", "loss")]:
            value = max(0.0, min(1.0, float(row[column]) if not pd.isna(row[column]) else 0.0))
            w = value * chart_w
            parts.append(f"<rect x=\"{x:.1f}\" y=\"{y + 4}\" width=\"{w:.1f}\" height=\"14\" class=\"{cls}\" />")
            x += w
        parts.append(f"<text x=\"{left + chart_w + 5}\" y=\"{y + 15}\" class=\"bar-value\">{pct(row['win_rate'])}</text>")
    parts.append("</svg>")
    return "".join(parts)


def svg_points(df: pd.DataFrame, x_col: str, y_col: str, x_min: float, x_max: float, y_min: float, y_max: float, left: float, top: float, width: float, height: float) -> str:
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return ""
    points = []
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)
    for _, row in df[[x_col, y_col]].dropna().iterrows():
        x = left + ((float(row[x_col]) - x_min) / x_span) * width
        y = top + height - ((float(row[y_col]) - y_min) / y_span) * height
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def trajectory_plot_svg(sample_row: dict, width=920, height=320) -> str:
    output_csv = sample_row.get("output_csv", "")
    df = read_trajectory_sample(output_csv)
    if df is None or df.empty:
        return "<p class=\"muted\">Could not read sampled trajectory.</p>"

    required = ["elapsed", "up_price", "down_price", "balance_after"]
    if any(column not in df.columns for column in required):
        return "<p class=\"muted\">Sampled trajectory is missing plot columns.</p>"

    for column in ["elapsed", "up_price", "down_price", "balance_after", "final_balance", "total_reward"]:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=["elapsed", "up_price", "down_price", "balance_after"]).sort_values("elapsed")
    if df.empty:
        return "<p class=\"muted\">Sampled trajectory has no plottable rows.</p>"

    action_col = "action_executed" if "action_executed" in df.columns else "action"
    if action_col not in df.columns:
        df[action_col] = "hold"

    if len(df) > 450:
        step = max(1, math.ceil(len(df) / 450))
        line_df = df.iloc[::step].copy()
        if line_df.index[-1] != df.index[-1]:
            line_df = pd.concat([line_df, df.tail(1)])
    else:
        line_df = df

    x_min = float(df["elapsed"].min())
    x_max = float(df["elapsed"].max())
    left = 54
    right = 18
    chart_w = width - left - right
    price_top = 48
    price_h = 120
    balance_top = 206
    balance_h = 78

    balance_values = df["balance_after"].dropna().tolist()
    final_values = df["final_balance"].dropna().tolist() if "final_balance" in df.columns else []
    balance_values.extend(final_values)
    balance_min = min(balance_values) if balance_values else 0.0
    balance_max = max(balance_values) if balance_values else 1.0
    if abs(balance_max - balance_min) < 1e-9:
        balance_min -= 1.0
        balance_max += 1.0
    else:
        pad = (balance_max - balance_min) * 0.08
        balance_min -= pad
        balance_max += pad

    up_points = svg_points(line_df, "elapsed", "up_price", x_min, x_max, 0.0, 1.0, left, price_top, chart_w, price_h)
    down_points = svg_points(line_df, "elapsed", "down_price", x_min, x_max, 0.0, 1.0, left, price_top, chart_w, price_h)
    balance_points = svg_points(line_df, "elapsed", "balance_after", x_min, x_max, balance_min, balance_max, left, balance_top, chart_w, balance_h)

    title = f"{sample_row.get('strategy_family', '')} | {Path(str(output_csv)).name}"
    reward = sample_row.get("total_reward", "")
    outcome = sample_row.get("final_outcome", "")
    subtitle = f"variant={sample_row.get('strategy_key', '')} reward={fmt(reward)} outcome={outcome}"

    parts = [
        f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"{html.escape(title)}\">",
        f"<text x=\"16\" y=\"22\" class=\"chart-title\">{html.escape(title[:96])}</text>",
        f"<text x=\"16\" y=\"40\" class=\"axis-label\">{html.escape(subtitle[:130])}</text>",
        f"<line x1=\"{left}\" y1=\"{price_top}\" x2=\"{left + chart_w}\" y2=\"{price_top}\" class=\"grid-line\" />",
        f"<line x1=\"{left}\" y1=\"{price_top + price_h / 2}\" x2=\"{left + chart_w}\" y2=\"{price_top + price_h / 2}\" class=\"grid-line\" />",
        f"<line x1=\"{left}\" y1=\"{price_top + price_h}\" x2=\"{left + chart_w}\" y2=\"{price_top + price_h}\" class=\"grid-line\" />",
        f"<text x=\"12\" y=\"{price_top + 5}\" class=\"axis-label\">1.0</text>",
        f"<text x=\"12\" y=\"{price_top + price_h / 2 + 4}\" class=\"axis-label\">0.5</text>",
        f"<text x=\"12\" y=\"{price_top + price_h + 4}\" class=\"axis-label\">0.0</text>",
        f"<polyline points=\"{up_points}\" class=\"line-up\" />",
        f"<polyline points=\"{down_points}\" class=\"line-down\" />",
        f"<line x1=\"{left}\" y1=\"{balance_top}\" x2=\"{left + chart_w}\" y2=\"{balance_top}\" class=\"grid-line\" />",
        f"<line x1=\"{left}\" y1=\"{balance_top + balance_h}\" x2=\"{left + chart_w}\" y2=\"{balance_top + balance_h}\" class=\"grid-line\" />",
        f"<text x=\"8\" y=\"{balance_top + 5}\" class=\"axis-label\">{fmt(balance_max, 1)}</text>",
        f"<text x=\"8\" y=\"{balance_top + balance_h + 4}\" class=\"axis-label\">{fmt(balance_min, 1)}</text>",
        f"<polyline points=\"{balance_points}\" class=\"line-balance\" />",
    ]

    if final_values:
        final_balance = float(final_values[-1])
        y_span = max(balance_max - balance_min, 1e-9)
        final_y = balance_top + balance_h - ((final_balance - balance_min) / y_span) * balance_h
        parts.append(f"<line x1=\"{left}\" y1=\"{final_y:.1f}\" x2=\"{left + chart_w}\" y2=\"{final_y:.1f}\" class=\"line-final\" />")
        parts.append(f"<text x=\"{left + chart_w - 150}\" y=\"{final_y - 5:.1f}\" class=\"axis-label\">final {fmt(final_balance, 2)}</text>")

    action_rows = df[df[action_col].fillna("hold").astype(str).ne("hold")].head(80)
    x_span = max(x_max - x_min, 1e-9)
    for _, row in action_rows.iterrows():
        action = str(row.get(action_col, ""))
        if action in {"buy_up", "sell_up"}:
            y_value = row.get("up_price")
            color_class = "marker-up"
        elif action in {"buy_down", "sell_down"}:
            y_value = row.get("down_price")
            color_class = "marker-down"
        else:
            continue
        if pd.isna(y_value):
            continue
        x = left + ((float(row["elapsed"]) - x_min) / x_span) * chart_w
        y = price_top + price_h - float(y_value) * price_h
        if action.startswith("sell"):
            parts.append(f"<text x=\"{x - 4:.1f}\" y=\"{y + 4:.1f}\" class=\"{color_class}\">x</text>")
        else:
            parts.append(f"<circle cx=\"{x:.1f}\" cy=\"{y:.1f}\" r=\"3.6\" class=\"{color_class}\" />")

    parts.extend([
        f"<text x=\"{left}\" y=\"{height - 14}\" class=\"legend-up\">UP</text>",
        f"<text x=\"{left + 42}\" y=\"{height - 14}\" class=\"legend-down\">DOWN</text>",
        f"<text x=\"{left + 104}\" y=\"{height - 14}\" class=\"legend-balance\">balance</text>",
        "</svg>",
    ])
    return "".join(parts)


def sample_plots_html(sample_rows: list[dict]) -> str:
    if not sample_rows:
        return "<p class=\"muted\">No sample plots selected.</p>"
    cards = []
    for row in sample_rows:
        cards.append(f"<div class=\"plot-card\">{trajectory_plot_svg(row)}</div>")
    return f"<div class=\"plot-grid\">{''.join(cards)}</div>"


def build_balance_path_rows(summary_df: pd.DataFrame) -> list[dict]:
    ok_df = summary_df[summary_df["status"].fillna("ok").eq("ok")].copy()
    if ok_df.empty or "strategy_key" not in ok_df.columns:
        return []

    for column in ["starting_balance", "final_balance", "strategy_market_no"]:
        if column in ok_df.columns:
            ok_df[column] = pd.to_numeric(ok_df[column], errors="coerce")

    rows = []
    for strategy_key, group in ok_df.groupby("strategy_key", dropna=False):
        ordered = group.copy()
        if "strategy_market_no" in ordered.columns and ordered["strategy_market_no"].notna().any():
            ordered = ordered.sort_values(["strategy_market_no", "market_file"], na_position="last")
        else:
            ordered = ordered.sort_values("market_file", na_position="last")

        points = []
        for idx, (_, row) in enumerate(ordered.iterrows(), start=1):
            start_balance = row.get("strategy_balance_before_market", row.get("starting_balance"))
            end_balance = row.get("strategy_balance_after_market", row.get("final_balance"))
            start_balance = pd.to_numeric(pd.Series([start_balance]), errors="coerce").iloc[0]
            end_balance = pd.to_numeric(pd.Series([end_balance]), errors="coerce").iloc[0]
            if pd.isna(start_balance) or pd.isna(end_balance):
                continue
            points.append({"x": idx - 0.45, "balance": float(start_balance), "kind": "start"})
            points.append({"x": idx, "balance": float(end_balance), "kind": "end"})

        if not points:
            continue

        rows.append({
            "strategy_key": strategy_key,
            "points": points,
            "market_count": int(len(ordered)),
            "first_balance": float(points[0]["balance"]),
            "last_balance": float(points[-1]["balance"]),
            "max_balance": max(point["balance"] for point in points),
            "min_balance": min(point["balance"] for point in points),
        })
    return rows


def balance_path_svg(balance_row: dict, width=620, height=190) -> str:
    points = balance_row.get("points", [])
    if not points:
        return "<p class=\"muted\">No balance path data.</p>"

    x_values = [point["x"] for point in points]
    y_values = [point["balance"] for point in points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if abs(y_max - y_min) < 1e-9:
        y_min -= 1.0
        y_max += 1.0
    else:
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad

    left = 44
    right = 16
    top = 34
    bottom = 28
    chart_w = width - left - right
    chart_h = height - top - bottom
    x_span = max(x_max - x_min, 1e-9)
    y_span = max(y_max - y_min, 1e-9)

    line_points = []
    for point in points:
        x = left + ((point["x"] - x_min) / x_span) * chart_w
        y = top + chart_h - ((point["balance"] - y_min) / y_span) * chart_h
        line_points.append(f"{x:.1f},{y:.1f}")

    title = str(balance_row.get("strategy_key", ""))
    subtitle = (
        f"markets={balance_row.get('market_count', 0)} "
        f"start={fmt(balance_row.get('first_balance'), 2)} "
        f"end={fmt(balance_row.get('last_balance'), 2)}"
    )
    return "".join([
        f"<svg viewBox=\"0 0 {width} {height}\" role=\"img\" aria-label=\"{html.escape(title)} balance path\">",
        f"<text x=\"12\" y=\"20\" class=\"chart-title\">{html.escape(title[:72])}</text>",
        f"<text x=\"12\" y=\"33\" class=\"axis-label\">{html.escape(subtitle)}</text>",
        f"<line x1=\"{left}\" y1=\"{top}\" x2=\"{left}\" y2=\"{top + chart_h}\" class=\"grid-line\" />",
        f"<line x1=\"{left}\" y1=\"{top + chart_h}\" x2=\"{left + chart_w}\" y2=\"{top + chart_h}\" class=\"grid-line\" />",
        f"<text x=\"6\" y=\"{top + 5}\" class=\"axis-label\">{fmt(y_max, 1)}</text>",
        f"<text x=\"6\" y=\"{top + chart_h + 4}\" class=\"axis-label\">{fmt(y_min, 1)}</text>",
        f"<polyline points=\"{' '.join(line_points)}\" class=\"line-balance\" />",
        f"<circle cx=\"{line_points[0].split(',')[0]}\" cy=\"{line_points[0].split(',')[1]}\" r=\"3\" class=\"marker-up\" />",
        f"<circle cx=\"{line_points[-1].split(',')[0]}\" cy=\"{line_points[-1].split(',')[1]}\" r=\"3\" class=\"marker-down\" />",
        "</svg>",
    ])


def balance_paths_html(balance_rows: list[dict]) -> str:
    if not balance_rows:
        return "<p class=\"muted\">No balance path data.</p>"
    cards = [f"<div class=\"plot-card\">{balance_path_svg(row)}</div>" for row in balance_rows]
    return f"<div class=\"plot-grid\">{''.join(cards)}</div>"


def render_report(
    run_folder,
    summary_df,
    stats_df,
    action_df,
    example_df,
    trajectory_examples,
    sample_plot_rows,
    family_stats_df,
    tp_stats_df,
    cheap_stats_df,
    size_stats_df,
    family_size_pivot_df,
    balance_path_rows,
) -> str:
    ok_count = int(summary_df["status"].fillna("ok").eq("ok").sum())
    failed_count = int(summary_df["status"].fillna("ok").ne("ok").sum())
    market_count = int(summary_df["market_file"].nunique()) if "market_file" in summary_df.columns else 0
    strategy_count = int(summary_df["strategy_key"].nunique())
    best = stats_df.iloc[0] if not stats_df.empty else {}

    stats_with_actions = stats_df.merge(action_df, on="strategy_key", how="left") if not stats_df.empty else stats_df
    stats_columns = [
        "strategy_key",
        "runs",
        "markets",
        "mean_reward",
        "median_reward",
        "total_reward",
        "win_rate",
        "loss_rate",
        "min_reward",
        "max_reward",
        "mean_final_balance",
        "hold_rows",
        "buy_up_rows",
        "buy_down_rows",
        "event_rows",
        "example_reasons",
    ]
    stats_formatters = {
        "mean_reward": fmt,
        "median_reward": fmt,
        "total_reward": fmt,
        "win_rate": pct,
        "loss_rate": pct,
        "min_reward": fmt,
        "max_reward": fmt,
        "mean_final_balance": fmt,
    }
    dimension_columns = [
        "strategies",
        "runs",
        "markets",
        "mean_reward",
        "median_reward",
        "total_reward",
        "mean_final_balance",
        "win_rate",
        "loss_rate",
        "min_reward",
        "max_reward",
        "reward_std",
    ]
    dimension_formatters = {
        "mean_reward": fmt,
        "median_reward": fmt,
        "total_reward": fmt,
        "mean_final_balance": fmt,
        "win_rate": pct,
        "loss_rate": pct,
        "min_reward": fmt,
        "max_reward": fmt,
        "reward_std": fmt,
    }
    example_columns = [
        "strategy_key",
        "example_type",
        "market_file",
        "final_outcome",
        "final_balance",
        "total_reward",
        "output_csv",
    ]
    trajectory_columns = [
        "strategy_key",
        "trajectory",
        "elapsed",
        "action",
        "reason",
        "balance_before",
        "balance_after",
        "final_balance",
        "total_reward",
    ]
    family_size_columns = ["strategy_family"] + [column for column in family_size_pivot_df.columns if column != "strategy_family"]
    family_size_formatters = {column: fmt for column in family_size_columns if column != "strategy_family"}

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Strategy Report - {html.escape(Path(run_folder).name)}</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f5f7fb;
      --panel: #ffffff;
      --text: #182230;
      --muted: #667085;
      --line: #d8dee9;
      --blue: #2563eb;
      --green: #16a34a;
      --red: #dc2626;
      --amber: #d97706;
    }}
    body {{ margin: 0; font-family: Arial, sans-serif; background: var(--bg); color: var(--text); }}
    header {{ padding: 28px 34px 18px; background: #111827; color: white; }}
    header h1 {{ margin: 0 0 8px; font-size: 26px; }}
    header .sub {{ color: #cbd5e1; }}
    main {{ padding: 24px 34px 40px; }}
    section {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 18px; margin-bottom: 18px; }}
    h2 {{ margin: 0 0 14px; font-size: 18px; }}
    .cards {{ display: grid; grid-template-columns: repeat(5, minmax(120px, 1fr)); gap: 12px; margin-bottom: 18px; }}
    .card {{ background: var(--panel); border: 1px solid var(--line); border-radius: 8px; padding: 14px; }}
    .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: .04em; }}
    .value {{ font-size: 22px; font-weight: 700; margin-top: 6px; }}
    table {{ width: 100%; border-collapse: collapse; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid var(--line); padding: 8px 9px; text-align: left; vertical-align: top; }}
    th {{ background: #f8fafc; position: sticky; top: 0; }}
    table.sortable th {{ cursor: pointer; user-select: none; white-space: nowrap; }}
    table.sortable th:hover {{ background: #eef4ff; }}
    .sort-indicator {{ color: var(--muted); font-size: 10px; margin-left: 5px; }}
    .table-wrap {{ overflow-x: auto; }}
    .muted {{ color: var(--muted); }}
    svg {{ min-width: 920px; width: 100%; height: auto; background: #fbfcff; border: 1px solid var(--line); border-radius: 8px; margin-bottom: 12px; }}
    .chart-scroll {{ overflow-x: auto; }}
    .chart-title {{ font: 700 15px Arial, sans-serif; fill: var(--text); }}
    .axis {{ stroke: #9aa4b2; stroke-width: 1; }}
    .axis-label {{ font: 12px Arial, sans-serif; fill: #344054; }}
    .bar-value {{ font: 12px Arial, sans-serif; fill: #344054; }}
    .bar.positive {{ fill: var(--green); }}
    .bar.negative {{ fill: var(--red); }}
    .win {{ fill: #16a34a; }}
    .loss {{ fill: #dc2626; }}
    .flat {{ fill: #d0d5dd; }}
    .plot-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 14px; }}
    .plot-card svg {{ margin: 0; min-width: 0; }}
    .grid-2 {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 16px; }}
    .grid-line {{ stroke: #d0d5dd; stroke-width: 1; }}
    .line-up {{ fill: none; stroke: #16a34a; stroke-width: 1.7; }}
    .line-down {{ fill: none; stroke: #dc2626; stroke-width: 1.7; }}
    .line-balance {{ fill: none; stroke: #2563eb; stroke-width: 1.7; }}
    .line-final {{ stroke: #111827; stroke-width: 1.2; stroke-dasharray: 5 4; }}
    .marker-up {{ fill: #16a34a; font: 700 13px Arial, sans-serif; }}
    .marker-down {{ fill: #dc2626; font: 700 13px Arial, sans-serif; }}
    .legend-up {{ fill: #16a34a; font: 700 12px Arial, sans-serif; }}
    .legend-down {{ fill: #dc2626; font: 700 12px Arial, sans-serif; }}
    .legend-balance {{ fill: #2563eb; font: 700 12px Arial, sans-serif; }}
  </style>
</head>
<body>
  <header>
    <h1>Strategy Performance Report</h1>
    <div class="sub">Run folder: {html.escape(str(run_folder))}</div>
  </header>
  <main>
    <div class="cards">
      <div class="card"><div class="label">Strategies</div><div class="value">{strategy_count}</div></div>
      <div class="card"><div class="label">Markets</div><div class="value">{market_count}</div></div>
      <div class="card"><div class="label">Completed</div><div class="value">{ok_count}</div></div>
      <div class="card"><div class="label">Failed</div><div class="value">{failed_count}</div></div>
      <div class="card"><div class="label">Best mean reward</div><div class="value">{html.escape(str(best.get('strategy_key', '')))}</div></div>
    </div>

    <section>
      <h2>Charts</h2>
      <div class="chart-scroll">{bar_chart_svg(stats_df, "mean_reward", "Mean reward by strategy")}</div>
      <div class="chart-scroll">{bar_chart_svg(stats_df, "total_reward", "Total reward by strategy")}</div>
      <div class="chart-scroll">{win_rate_svg(stats_df)}</div>
    </section>

    <section>
      <h2>Parameter Analysis</h2>
      <p class="muted">These views collapse individual strategy names into the main knobs: family, take-profit, cheap-entry threshold, and sizing mode.</p>
      <div class="grid-2">
        <div>
          <h3>Mean Reward By Family</h3>
          <div class="chart-scroll">{bar_chart_svg(family_stats_df.rename(columns={"strategy_family": "strategy_key"}), "mean_reward", "Mean reward by family", width=980, height=240)}</div>
        </div>
        <div>
          <h3>Mean Reward By Size</h3>
          <div class="chart-scroll">{bar_chart_svg(size_stats_df.rename(columns={"size_code": "strategy_key"}), "mean_reward", "Mean reward by size", width=980, height=220)}</div>
        </div>
      </div>
      <div class="grid-2">
        <div class="table-wrap">
          <h3>Family Summary</h3>
          {table_html(family_stats_df, ["strategy_family"] + dimension_columns, dimension_formatters, sortable=True)}
        </div>
        <div class="table-wrap">
          <h3>Size Summary</h3>
          {table_html(size_stats_df, ["size_code"] + dimension_columns, dimension_formatters, sortable=True)}
        </div>
      </div>
      <div class="grid-2">
        <div class="table-wrap">
          <h3>Take Profit Summary</h3>
          {table_html(tp_stats_df, ["tp_code"] + dimension_columns, dimension_formatters, sortable=True)}
        </div>
        <div class="table-wrap">
          <h3>Cheap Threshold Summary</h3>
          {table_html(cheap_stats_df, ["cheap_code"] + dimension_columns, dimension_formatters, sortable=True)}
        </div>
      </div>
      <div class="table-wrap">
        <h3>Family vs Size Mean Outcome</h3>
        <p class="muted">This makes it easier to see whether a family is winning because of logic or because of sizing.</p>
        {table_html(family_size_pivot_df, family_size_columns, family_size_formatters, sortable=True)}
      </div>
    </section>

    <section>
      <h2>Balance Paths By Strategy</h2>
      <p class="muted">Each chart samples balance at the start and end of every market in sequence for that strategy. In compound mode this shows bankroll carryover directly.</p>
      {balance_paths_html(balance_path_rows)}
    </section>

    <section>
      <h2>Random Sample Trajectory Plots</h2>
      <p class="muted">One random trajectory per strategy family. Generated parameter variations are grouped together.</p>
      {sample_plots_html(sample_plot_rows)}
    </section>

    <section>
      <h2>Strategy Stats</h2>
      <div class="table-wrap">
        {table_html(stats_with_actions, stats_columns, stats_formatters, table_id="strategy-stats-table", sortable=True)}
      </div>
    </section>

    <section>
      <h2>Best And Worst Market Rows</h2>
      <div class="table-wrap">
        {table_html(example_df, example_columns, {"final_balance": fmt, "total_reward": fmt})}
      </div>
    </section>

    <section>
      <h2>Example Trajectory Rows</h2>
      <p class="muted">These are sampled action rows from each strategy's trajectory CSVs.</p>
      <div class="table-wrap">
        {table_html(trajectory_examples, trajectory_columns, {
            "elapsed": fmt,
            "balance_before": fmt,
            "balance_after": fmt,
            "final_balance": fmt,
            "total_reward": fmt,
        })}
      </div>
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
      if (typeof a === "number" && typeof b === "number") {{
        return a - b;
      }}
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


def main():
    args = parse_args()
    run_folder = Path(args.run_folder) if args.run_folder else newest_run_folder()
    generate_report(
        run_folder=run_folder,
        output_path=Path(args.output) if args.output else None,
        examples_per_strategy=args.examples_per_strategy,
        max_trajectory_files_per_strategy=args.max_trajectory_files_per_strategy,
        sample_plot_seed=args.sample_plot_seed,
        max_sample_plots=args.max_sample_plots,
        include_sample_plots=not args.no_sample_plots,
    )


def generate_report(
    *,
    run_folder: str | Path,
    output_path: str | Path | None = None,
    examples_per_strategy: int = 3,
    max_trajectory_files_per_strategy: int = 5,
    sample_plot_seed: int = 42,
    max_sample_plots: int = 50,
    include_sample_plots: bool = True,
) -> Path:
    run_folder = Path(run_folder)
    output_path = Path(output_path) if output_path else run_folder / REPORT_FILENAME

    summary_df = enrich_strategy_dimensions(load_summary(run_folder))
    stats_df = build_strategy_stats(summary_df)
    action_df = build_action_stats(summary_df, max_trajectory_files_per_strategy)
    example_df = pick_example_rows(summary_df, examples_per_strategy)
    trajectory_examples = pick_trajectory_examples(summary_df, max_trajectory_files_per_strategy)
    family_stats_df = build_dimension_stats(summary_df, "strategy_family")
    tp_stats_df = build_dimension_stats(summary_df, "tp_code")
    cheap_stats_df = build_dimension_stats(summary_df, "cheap_code")
    size_stats_df = build_dimension_stats(summary_df, "size_code")
    family_size_pivot_df = build_family_size_pivot(summary_df)
    balance_path_rows = build_balance_path_rows(summary_df)
    sample_plot_rows = [] if not include_sample_plots else pick_sample_plot_rows(
        summary_df,
        seed=sample_plot_seed,
        max_plots=max_sample_plots,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        render_report(
            run_folder,
            summary_df,
            stats_df,
            action_df,
            example_df,
            trajectory_examples,
            sample_plot_rows,
            family_stats_df,
            tp_stats_df,
            cheap_stats_df,
            size_stats_df,
            family_size_pivot_df,
            balance_path_rows,
        ),
        encoding="utf-8",
    )

    print(f"Run folder: {run_folder}")
    print(f"Strategies: {summary_df['strategy_key'].nunique()}")
    print(f"Rows:       {len(summary_df)}")
    print(f"Plots:      {len(sample_plot_rows)}")
    print(f"Report:     {output_path}")
    return output_path


if __name__ == "__main__":
    main()
