import argparse
import csv
from pathlib import Path


INPUT_DIR = Path("data")
OUTPUT_DIR = Path("training_data")
CSV_PATTERN = "btc-updown-5m-*.csv"


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


def parse_float(value):
    if value is None:
        return None
    value = str(value).strip()
    if value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def format_csv_value(value):
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.15g}"
    return value


def latest_value_at_or_before(series, cutoff_time):
    for timestamp, value in reversed(series):
        if timestamp <= cutoff_time:
            return value
    return None


def raw_change(series, now_time, window_s):
    if not series:
        return ""
    current = series[-1][1]
    previous = latest_value_at_or_before(series, now_time - window_s)
    if current is None or previous is None:
        return ""
    return current - previous


def decimal_return(series, now_time, window_s):
    if not series:
        return ""
    current = series[-1][1]
    previous = latest_value_at_or_before(series, now_time - window_s)
    if current is None or previous in (None, 0):
        return ""
    return (current - previous) / previous


def short_return_volatility(series, now_time, window_s):
    window = [(ts, price) for ts, price in series if ts >= now_time - window_s and price not in (None, 0)]
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


def choose_btc_series(chainlink_series, binance_series):
    if chainlink_series:
        return chainlink_series
    return binance_series


def build_feature_row(input_row, now_time, up_series, down_series, btc_binance_series, btc_chainlink_series):
    up_price = up_series[-1][1] if up_series else None
    down_price = down_series[-1][1] if down_series else None
    btc_binance = btc_binance_series[-1][1] if btc_binance_series else None
    btc_chainlink = btc_chainlink_series[-1][1] if btc_chainlink_series else None
    btc_price = btc_chainlink if btc_chainlink is not None else btc_binance
    btc_series = choose_btc_series(btc_chainlink_series, btc_binance_series)

    elapsed = parse_float(input_row.get("elapsed"))
    price_to_beat = parse_float(input_row.get("price_to_beat"))

    btc_distance_to_beat_pct = ""
    market_confidence_gap = ""

    if btc_price is not None and price_to_beat not in (None, 0):
        btc_distance_to_beat = btc_price - price_to_beat
        btc_distance_to_beat_pct = btc_distance_to_beat / price_to_beat
        btc_above_price_to_beat = 1 if btc_price > price_to_beat else 0
        if up_price is not None:
            market_confidence_gap = btc_above_price_to_beat - up_price

    return {
        "time_fraction_elapsed": elapsed / 300.0 if elapsed is not None else "",
        "up_price": up_price,
        "up_price_change_0_5s": raw_change(up_series, now_time, 0.5),
        "up_price_change_1s": raw_change(up_series, now_time, 1.0),
        "up_price_change_5s": raw_change(up_series, now_time, 5.0),
        "up_price_change_15s": raw_change(up_series, now_time, 15.0),
        "up_price_change_30s": raw_change(up_series, now_time, 30.0),
        "btc_return_0_5s": decimal_return(btc_series, now_time, 0.5),
        "btc_return_1s": decimal_return(btc_series, now_time, 1.0),
        "btc_return_5s": decimal_return(btc_series, now_time, 5.0),
        "btc_return_15s": decimal_return(btc_series, now_time, 15.0),
        "btc_return_30s": decimal_return(btc_series, now_time, 30.0),
        "btc_return_60s": decimal_return(btc_series, now_time, 60.0),
        "btc_volatility_5s": short_return_volatility(btc_series, now_time, 5.0),
        "btc_volatility_15s": short_return_volatility(btc_series, now_time, 15.0),
        "btc_volatility_30s": short_return_volatility(btc_series, now_time, 30.0),
        "btc_volatility_60s": short_return_volatility(btc_series, now_time, 60.0),
        "btc_distance_to_beat_pct": btc_distance_to_beat_pct,
        "market_confidence_gap": market_confidence_gap,
    }


def transform_file(input_path, output_path):
    up_series = []
    down_series = []
    btc_binance_series = []
    btc_chainlink_series = []
    rows_written = 0
    rows_skipped = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with input_path.open("r", newline="", encoding="utf-8-sig") as source, output_path.open(
        "w", newline="", encoding="utf-8"
    ) as target:
        reader = csv.DictReader(source)
        writer = csv.DictWriter(target, fieldnames=MARKET_CSV_HEADER)
        writer.writeheader()

        for input_row in reader:
            up_price = parse_float(input_row.get("up_price"))
            down_price = parse_float(input_row.get("down_price"))
            unix_time = parse_float(input_row.get("unix_time"))
            elapsed = parse_float(input_row.get("elapsed"))

            if up_price is None or down_price is None or unix_time is None:
                rows_skipped += 1
                continue

            now_time = unix_time
            up_series.append((now_time, up_price))
            down_series.append((now_time, down_price))

            btc_binance = parse_float(input_row.get("btc_binance"))
            btc_chainlink = parse_float(input_row.get("btc_chainlink"))
            if btc_binance is not None:
                btc_binance_series.append((now_time, btc_binance))
            if btc_chainlink is not None:
                btc_chainlink_series.append((now_time, btc_chainlink))

            if elapsed is None:
                input_row["elapsed"] = ""

            feature_row = build_feature_row(
                input_row,
                now_time,
                up_series,
                down_series,
                btc_binance_series,
                btc_chainlink_series,
            )
            writer.writerow({column: format_csv_value(feature_row.get(column)) for column in MARKET_CSV_HEADER})
            rows_written += 1

    return rows_written, rows_skipped


def transform_folder(input_dir=INPUT_DIR, output_dir=OUTPUT_DIR, pattern=CSV_PATTERN):
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    csv_files = sorted(input_dir.glob(pattern))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {input_dir} using pattern {pattern!r}")

    total_written = 0
    total_skipped = 0
    failed = []

    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Files:  {len(csv_files)}")
    print()

    for input_path in csv_files:
        output_path = output_dir / input_path.name
        try:
            rows_written, rows_skipped = transform_file(input_path, output_path)
            total_written += rows_written
            total_skipped += rows_skipped
            print(f"{input_path.name}: wrote {rows_written} rows, skipped {rows_skipped}")
        except PermissionError as e:
            failed.append((input_path.name, str(e)))
            print(f"{input_path.name}: FAILED permission error: {e}")

    print()
    print("Done.")
    print(f"Total rows written: {total_written}")
    print(f"Total rows skipped: {total_skipped}")
    if failed:
        print(f"Failed files: {len(failed)}")
        for filename, error in failed:
            print(f"  {filename}: {error}")


def main():
    parser = argparse.ArgumentParser(
        description="Convert recorded market CSVs to market-only feature CSVs for training input."
    )
    parser.add_argument("--input-dir", default=str(INPUT_DIR), help="Folder containing source market CSVs.")
    parser.add_argument("--output-dir", default=str(OUTPUT_DIR), help="Folder to write transformed CSVs.")
    parser.add_argument("--pattern", default=CSV_PATTERN, help="Glob pattern for source CSV files.")
    args = parser.parse_args()

    transform_folder(args.input_dir, args.output_dir, args.pattern)


if __name__ == "__main__":
    main()
