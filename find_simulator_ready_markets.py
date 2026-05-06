import argparse
import csv
import shutil
from pathlib import Path


DEFAULT_INPUT_DIRS = [Path("data"), Path("data_uncleaned")]
DEFAULT_OUTPUT_DIR = Path("simulator_ready_markets")
DEFAULT_REJECT_LOG = Path("simulator_ready_market_rejections.csv")
CSV_PATTERN = "btc-updown-5m-*.csv"
DEFAULT_MAX_FIRST_ELAPSED_S = 15.0

# Minimum schema needed by simulator.replay.load_market_ticks and outcome inference.
REQUIRED_COLUMNS = {
    "timestamp",
    "unix_time",
    "seconds_left",
    "elapsed",
    "up_price",
    "down_price",
    "btc_binance",
    "btc_chainlink",
    "price_to_beat",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy simulator-ready BTC market CSVs from data folders into one flat folder."
    )
    parser.add_argument(
        "--input-dir",
        action="append",
        dest="input_dirs",
        help="Input folder to scan recursively. Can be passed more than once.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Folder to copy ready market CSVs into. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--reject-log",
        default=str(DEFAULT_REJECT_LOG),
        help=f"CSV path for rejected file reasons. Default: {DEFAULT_REJECT_LOG}",
    )
    parser.add_argument(
        "--max-first-elapsed",
        type=float,
        default=DEFAULT_MAX_FIRST_ELAPSED_S,
        help=f"Maximum allowed first elapsed value. Default: {DEFAULT_MAX_FIRST_ELAPSED_S}",
    )
    return parser.parse_args()


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


def file_has_nul_bytes(csv_path: Path) -> bool:
    with csv_path.open("rb") as f:
        return b"\x00" in f.read(8192)


def read_rows(csv_path: Path):
    if file_has_nul_bytes(csv_path):
        raise ValueError("file contains NUL bytes, likely corrupted or wrong encoding")

    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            fieldnames = set(reader.fieldnames or [])
            rows = list(reader)
    except UnicodeDecodeError as e:
        raise ValueError(f"could not decode as UTF-8 CSV: {e}") from e
    except csv.Error as e:
        raise ValueError(f"CSV parse error: {e}") from e

    return fieldnames, rows


def first_valid_elapsed(rows):
    for row in rows:
        elapsed = parse_float(row.get("elapsed"))
        if elapsed is not None:
            return elapsed
    return None


def has_replayable_price_row(rows):
    for row in rows:
        if parse_float(row.get("up_price")) is not None and parse_float(row.get("down_price")) is not None:
            return True
    return False


def has_outcome_inputs(rows):
    has_btc = False
    has_price_to_beat = False

    for row in rows:
        btc = parse_float(row.get("btc_chainlink"))
        if btc is None:
            btc = parse_float(row.get("btc_binance"))
        if btc is not None:
            has_btc = True

        if parse_float(row.get("price_to_beat")) is not None:
            has_price_to_beat = True

        if has_btc and has_price_to_beat:
            return True

    return False


def simulator_ready(csv_path: Path, max_first_elapsed: float):
    fieldnames, rows = read_rows(csv_path)

    missing = REQUIRED_COLUMNS - fieldnames
    if missing:
        return False, f"missing required columns: {sorted(missing)}"

    if not rows:
        return False, "no data rows"

    first_elapsed = first_valid_elapsed(rows)
    if first_elapsed is None:
        return False, "no valid elapsed value"
    if first_elapsed > max_first_elapsed:
        return False, f"first elapsed too late: {first_elapsed:.2f}s > {max_first_elapsed:.2f}s"

    if not has_replayable_price_row(rows):
        return False, "no row with valid up_price and down_price"

    if not has_outcome_inputs(rows):
        return False, "missing valid BTC price or price_to_beat values for outcome inference"

    return True, f"ready: first elapsed={first_elapsed:.2f}s"


def discover_csvs(input_dirs):
    csv_paths = []
    for input_dir in input_dirs:
        if not input_dir.exists():
            continue
        csv_paths.extend(sorted(input_dir.rglob(CSV_PATTERN)))
    return csv_paths


def write_reject_log(reject_log_path, reject_rows):
    with reject_log_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["source", "filename", "reason"])
        writer.writeheader()
        writer.writerows(reject_rows)


def main():
    args = parse_args()
    input_dirs = [Path(value) for value in args.input_dirs] if args.input_dirs else DEFAULT_INPUT_DIRS
    output_dir = Path(args.output_dir)
    reject_log_path = Path(args.reject_log)

    csv_paths = discover_csvs(input_dirs)
    output_dir.mkdir(parents=True, exist_ok=True)

    copied_names = set()
    ready_count = 0
    rejected_rows = []
    duplicate_count = 0

    print(f"Scanning:      {', '.join(str(path) for path in input_dirs)}")
    print(f"Files found:   {len(csv_paths)}")
    print(f"Output folder: {output_dir}")
    print(f"Elapsed limit: {args.max_first_elapsed:.2f}s")
    print()

    for csv_path in csv_paths:
        try:
            ok, reason = simulator_ready(csv_path, args.max_first_elapsed)
        except Exception as e:
            ok = False
            reason = f"unexpected error: {type(e).__name__}: {e}"

        if not ok:
            rejected_rows.append({
                "source": str(csv_path),
                "filename": csv_path.name,
                "reason": reason,
            })
            continue

        if csv_path.name in copied_names:
            duplicate_count += 1
            rejected_rows.append({
                "source": str(csv_path),
                "filename": csv_path.name,
                "reason": "duplicate ready market filename already copied",
            })
            continue

        shutil.copy2(csv_path, output_dir / csv_path.name)
        copied_names.add(csv_path.name)
        ready_count += 1

    write_reject_log(reject_log_path, rejected_rows)

    print("Done.")
    print(f"Ready copied:  {ready_count}")
    print(f"Rejected:      {len(rejected_rows) - duplicate_count}")
    print(f"Duplicates:    {duplicate_count}")
    print(f"Reject log:    {reject_log_path}")


if __name__ == "__main__":
    main()
