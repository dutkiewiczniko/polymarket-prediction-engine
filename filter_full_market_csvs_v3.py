from pathlib import Path
import csv
import shutil


UNCLEANED_DIR = Path("data_uncleaned")
CLEAN_DIR = Path("data")
REJECT_LOG_PATH = Path("rejected_market_files.csv")

# The CSV will usually not start at exactly 0.00 and 300.00.
# Example good file: elapsed = 0.24, seconds_left = 299.76
MAX_START_ELAPSED_S = 2.0
MIN_SECONDS_LEFT_S = 298.0

CSV_PATTERN = "btc-updown-5m-*.csv"

# Set to True if you want to move files instead of copying them.
MOVE_FILES = False

# Minimum schema needed for the replay engine.
# We do not need to check every event/order column, because replay ignores those for now.
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


def file_has_nul_bytes(csv_path: Path) -> bool:
    """Detect files that contain NUL bytes.

    The Python csv module crashes on these with:
        _csv.Error: line contains NUL

    We reject these files rather than letting the whole cleaning run fail.
    """
    with csv_path.open("rb") as f:
        sample = f.read(8192)
    return b"\x00" in sample


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


def read_header(csv_path: Path):
    """Read CSV header safely.

    Raises a friendly ValueError if the file is not a usable CSV.
    """
    if file_has_nul_bytes(csv_path):
        raise ValueError("file contains NUL bytes, likely corrupted or wrong encoding")

    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)
            return set(reader.fieldnames or [])
    except UnicodeDecodeError as e:
        raise ValueError(f"could not decode as UTF-8 CSV: {e}") from e
    except csv.Error as e:
        raise ValueError(f"CSV parse error while reading header: {e}") from e


def schema_is_supported(csv_path: Path):
    try:
        columns = read_header(csv_path)
    except ValueError as e:
        return False, str(e)

    missing = REQUIRED_COLUMNS - columns

    if missing:
        return False, f"old/unsupported schema, missing columns: {sorted(missing)}"

    return True, "schema ok"


def first_valid_time_row(csv_path: Path):
    """Return the first row that has usable elapsed and seconds_left values."""

    if file_has_nul_bytes(csv_path):
        return None, "file contains NUL bytes, likely corrupted or wrong encoding"

    try:
        with csv_path.open("r", newline="", encoding="utf-8-sig") as f:
            reader = csv.DictReader(f)

            for row in reader:
                elapsed = parse_float(row.get("elapsed"))
                seconds_left = parse_float(row.get("seconds_left"))

                if elapsed is None or seconds_left is None:
                    continue

                return {
                    "elapsed": elapsed,
                    "seconds_left": seconds_left,
                }, ""

    except UnicodeDecodeError as e:
        return None, f"could not decode as UTF-8 CSV: {e}"
    except csv.Error as e:
        return None, f"CSV parse error while reading rows: {e}"

    return None, "no valid elapsed/seconds_left row found"


def starts_at_market_beginning(csv_path: Path):
    first_row, error = first_valid_time_row(csv_path)

    if error:
        return False, error

    elapsed = first_row["elapsed"]
    seconds_left = first_row["seconds_left"]

    starts_near_zero = elapsed <= MAX_START_ELAPSED_S
    starts_near_300_left = seconds_left >= MIN_SECONDS_LEFT_S

    if starts_near_zero and starts_near_300_left:
        return True, f"start ok: elapsed={elapsed:.2f}, seconds_left={seconds_left:.2f}"

    return False, f"bad start: elapsed={elapsed:.2f}, seconds_left={seconds_left:.2f}"


def is_accepted_market_file(csv_path: Path):
    schema_ok, schema_message = schema_is_supported(csv_path)

    if not schema_ok:
        return False, schema_message

    start_ok, start_message = starts_at_market_beginning(csv_path)

    if not start_ok:
        return False, start_message

    return True, f"{schema_message}; {start_message}"


def main():
    if not UNCLEANED_DIR.exists():
        print(f"Folder not found: {UNCLEANED_DIR}")
        return

    CLEAN_DIR.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(UNCLEANED_DIR.rglob(CSV_PATTERN))

    if not csv_files:
        print(f"No files found in {UNCLEANED_DIR} matching {CSV_PATTERN!r}")
        return

    accepted = 0
    rejected = 0
    reject_rows = []

    print(f"Scanning {len(csv_files)} files from {UNCLEANED_DIR}")
    print(f"Copying accepted files to {CLEAN_DIR}")
    print()

    for csv_path in csv_files:
        source_label = str(csv_path.relative_to(UNCLEANED_DIR))

        try:
            ok, message = is_accepted_market_file(csv_path)
        except Exception as e:
            ok = False
            message = f"unexpected error: {type(e).__name__}: {e}"

        if ok:
            destination = CLEAN_DIR / csv_path.name

            if MOVE_FILES:
                shutil.move(str(csv_path), str(destination))
                action = "moved"
            else:
                shutil.copy2(csv_path, destination)
                action = "copied"

            accepted += 1
            print(f"[ACCEPTED] {source_label} -> {action} | {message}")
        else:
            rejected += 1
            reject_rows.append({
                "filename": source_label,
                "reason": message,
            })
            print(f"[REJECTED] {source_label} | {message}")

    if reject_rows:
        with REJECT_LOG_PATH.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["filename", "reason"])
            writer.writeheader()
            writer.writerows(reject_rows)

    print()
    print("Done.")
    print(f"Accepted: {accepted}")
    print(f"Rejected: {rejected}")
    print(f"Reject log: {REJECT_LOG_PATH}")


if __name__ == "__main__":
    main()
