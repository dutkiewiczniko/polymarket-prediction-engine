import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd


DEFAULT_TRAJECTORIES_FOLDER = Path("runs/first_batch_test/trajectories")
DEFAULT_OUTPUT_PATH = Path("datasets/training_dataset_v1.csv")

LOOKBACK_WINDOWS = [0.5, 1.0, 5.0, 15.0, 30.0]
BTC_RETURN_WINDOWS = [0.5, 1.0, 5.0, 15.0, 30.0, 60.0]
BTC_VOLATILITY_WINDOWS = [5.0, 15.0, 30.0, 60.0]

DEBUG_COLUMNS = [
    "market_id",
    "trajectory_file",
    "strategy_run_name",
    "timestamp",
    "unix_time",
    "row_index",
    "action",
    "reason",
    "final_outcome",
    "final_balance",
    "total_reward",
]

FEATURE_COLUMNS = [
    "elapsed",
    "seconds_left",
    "time_fraction_elapsed",
    "up_price",
    "down_price",
    "up_minus_down",
    "up_price_change_0_5s",
    "up_price_change_1s",
    "up_price_change_5s",
    "up_price_change_15s",
    "up_price_change_30s",
    "down_price_change_0_5s",
    "down_price_change_1s",
    "down_price_change_5s",
    "down_price_change_15s",
    "down_price_change_30s",
    "btc_price",
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
    "price_to_beat",
    "btc_distance_to_beat",
    "btc_distance_to_beat_pct",
    "btc_above_price_to_beat",
    "direction_signal",
    "market_confidence_gap",
    "abs_market_confidence_gap",
    "cash_before",
    "up_tokens_before",
    "down_tokens_before",
    "balance_before",
    "up_position_value",
    "down_position_value",
    "position_value",
    "position_exposure_pct",
    "net_position_tokens",
    "net_position_value",
    "action_hold",
    "action_buy_up",
    "action_buy_down",
]

OUTPUT_COLUMNS = DEBUG_COLUMNS + FEATURE_COLUMNS + ["target_reward_to_go"]
NUMERIC_SOURCE_COLUMNS = [
    "elapsed",
    "seconds_left",
    "up_price",
    "down_price",
    "btc_price",
    "btc_chainlink",
    "btc_binance",
    "price_to_beat",
    "cash_before",
    "up_tokens_before",
    "down_tokens_before",
    "balance_before",
    "final_balance",
    "total_reward",
    "unix_time",
]
ESSENTIAL_COLUMNS = [
    "elapsed",
    "seconds_left",
    "up_price",
    "down_price",
    "btc_price",
    "price_to_beat",
    "balance_before",
    "final_balance",
]
ACTION_COLUMNS = ["action_hold", "action_buy_up", "action_buy_down"]
ACTION_VALUES = {"hold", "buy_up", "buy_down"}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Build one combined offline ML training dataset from simulator trajectory CSVs."
    )
    parser.add_argument("--trajectories-folder", help="Folder containing simulator trajectory CSVs.")
    parser.add_argument("--output", help="Output CSV path.")
    return parser.parse_args()


def resolve_path(cli_value, prompt, default):
    if cli_value:
        return Path(cli_value)

    value = input(f"{prompt} [{default}]: ").strip()
    return Path(value) if value else default


def safe_ratio(numerator, denominator):
    if pd.isna(numerator) or pd.isna(denominator) or denominator == 0:
        return math.nan
    return numerator / denominator


def window_suffix(window):
    return str(window).replace(".", "_").replace("_0", "")


def infer_ids(csv_path, trajectories_folder):
    stem = csv_path.stem

    if "__" in stem:
        parts = stem.split("__")
        market_id = parts[0]
        strategy_run_name = "__".join(parts[1:]) if len(parts) > 1 else ""
        return market_id, strategy_run_name

    market_id = stem
    try:
        relative = csv_path.relative_to(trajectories_folder)
    except ValueError:
        relative = csv_path

    strategy_run_name = ""
    if len(relative.parts) >= 2:
        strategy_run_name = relative.parts[0]

    return market_id, strategy_run_name


def latest_at_or_before(elapsed_values, value_values, cutoff):
    matches = elapsed_values <= cutoff
    if not matches.any():
        return math.nan

    index = matches[matches].index[-1]
    return value_values.loc[index]


def raw_change_at(row, df, value_column, window):
    then = latest_at_or_before(df["elapsed"], df[value_column], row["elapsed"] - window)
    if pd.isna(row[value_column]) or pd.isna(then):
        return 0.0
    return row[value_column] - then


def decimal_return_at(row, df, value_column, window):
    then = latest_at_or_before(df["elapsed"], df[value_column], row["elapsed"] - window)
    if pd.isna(row[value_column]) or pd.isna(then) or then == 0:
        return 0.0
    return (row[value_column] - then) / then


def btc_volatility_at(row, df, window):
    window_df = df[(df["elapsed"] >= row["elapsed"] - window) & (df["elapsed"] <= row["elapsed"])]
    values = window_df["btc_price"].dropna()
    if len(values) < 3:
        return 0.0

    returns = values.pct_change().replace([math.inf, -math.inf], pd.NA).dropna()
    if len(returns) < 2:
        return 0.0

    return float(returns.std(ddof=0))


def add_lookback_features(df):
    elapsed = df["elapsed"].to_numpy(dtype=float)
    up_price = df["up_price"].to_numpy(dtype=float)
    down_price = df["down_price"].to_numpy(dtype=float)
    btc_price = df["btc_price"].to_numpy(dtype=float)

    for window in LOOKBACK_WINDOWS:
        suffix = window_suffix(window)
        previous_indexes = np.searchsorted(elapsed, elapsed - window, side="right") - 1
        valid = previous_indexes >= 0

        up_changes = np.zeros(len(df), dtype=float)
        down_changes = np.zeros(len(df), dtype=float)
        up_changes[valid] = up_price[valid] - up_price[previous_indexes[valid]]
        down_changes[valid] = down_price[valid] - down_price[previous_indexes[valid]]
        df[f"up_price_change_{suffix}s"] = up_changes
        df[f"down_price_change_{suffix}s"] = down_changes

    for window in BTC_RETURN_WINDOWS:
        suffix = window_suffix(window)
        previous_indexes = np.searchsorted(elapsed, elapsed - window, side="right") - 1
        valid = (previous_indexes >= 0) & (btc_price[previous_indexes] != 0)

        returns = np.zeros(len(df), dtype=float)
        returns[valid] = (btc_price[valid] - btc_price[previous_indexes[valid]]) / btc_price[previous_indexes[valid]]
        df[f"btc_return_{suffix}s"] = returns

    short_returns = np.zeros(len(df), dtype=float)
    if len(df) > 1:
        previous_btc = btc_price[:-1]
        current_btc = btc_price[1:]
        valid_previous = previous_btc != 0
        short_returns[1:][valid_previous] = (current_btc[valid_previous] - previous_btc[valid_previous]) / previous_btc[valid_previous]

    valid_return = np.isfinite(short_returns)
    if len(valid_return):
        valid_return[0] = False
    return_values = np.where(valid_return, short_returns, 0.0)
    prefix_sum = np.concatenate([[0.0], np.cumsum(return_values)])
    prefix_sq_sum = np.concatenate([[0.0], np.cumsum(return_values * return_values)])
    prefix_count = np.concatenate([[0], np.cumsum(valid_return.astype(int))])

    for window in BTC_VOLATILITY_WINDOWS:
        suffix = window_suffix(window)
        start_indexes = np.searchsorted(elapsed, elapsed - window, side="left")
        end_indexes = np.arange(len(df)) + 1

        counts = prefix_count[end_indexes] - prefix_count[start_indexes]
        sums = prefix_sum[end_indexes] - prefix_sum[start_indexes]
        sq_sums = prefix_sq_sum[end_indexes] - prefix_sq_sum[start_indexes]

        volatility = np.zeros(len(df), dtype=float)
        enough = counts >= 2
        means = np.zeros(len(df), dtype=float)
        means[enough] = sums[enough] / counts[enough]
        variances = np.zeros(len(df), dtype=float)
        variances[enough] = (sq_sums[enough] / counts[enough]) - (means[enough] * means[enough])
        variances = np.maximum(variances, 0.0)
        volatility[enough] = np.sqrt(variances[enough])
        df[f"btc_volatility_{suffix}s"] = volatility

    return df


def normalise_action(value):
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def prepare_source_frame(csv_path):
    df = pd.read_csv(csv_path)

    for column in NUMERIC_SOURCE_COLUMNS:
        if column in df.columns:
            df[column] = pd.to_numeric(df[column], errors="coerce")

    if "btc_price" not in df.columns:
        df["btc_price"] = pd.NA

    chainlink = df["btc_chainlink"] if "btc_chainlink" in df.columns else pd.Series(pd.NA, index=df.index)
    binance = df["btc_binance"] if "btc_binance" in df.columns else pd.Series(pd.NA, index=df.index)
    df["btc_price"] = df["btc_price"].where(df["btc_price"].notna(), chainlink)
    df["btc_price"] = df["btc_price"].where(df["btc_price"].notna(), binance)
    df["btc_price"] = pd.to_numeric(df["btc_price"], errors="coerce")

    if "action" not in df.columns:
        df["action"] = ""
    if "reason" not in df.columns:
        df["reason"] = ""

    for column in ["timestamp", "final_outcome"]:
        if column not in df.columns:
            df[column] = ""
    if "unix_time" not in df.columns:
        df["unix_time"] = pd.NA
    if "total_reward" not in df.columns:
        df["total_reward"] = pd.NA
    for column in ["cash_before", "up_tokens_before", "down_tokens_before"]:
        if column not in df.columns:
            df[column] = 0.0
        df[column] = pd.to_numeric(df[column], errors="coerce").fillna(0.0)

    df["row_index"] = range(len(df))
    df["action"] = df["action"].map(normalise_action)
    df = df.sort_values("elapsed", kind="mergesort").reset_index(drop=True)
    return df


def build_rows_for_file(csv_path, trajectories_folder):
    source = prepare_source_frame(csv_path)
    market_id, strategy_run_name = infer_ids(csv_path, trajectories_folder)

    for column in ESSENTIAL_COLUMNS:
        if column not in source.columns:
            return [], f"missing required column: {column}"

    source = source[source["action"].isin(ACTION_VALUES)].copy()
    source = source.dropna(subset=ESSENTIAL_COLUMNS)
    source = source[source["price_to_beat"] != 0]
    source = source[source["balance_before"] != 0]

    if source.empty:
        return [], "no rows with complete essential values and known actions"

    source = add_lookback_features(source.reset_index(drop=True))
    source["time_fraction_elapsed"] = source["elapsed"] / 300.0
    source["up_minus_down"] = source["up_price"] - source["down_price"]
    source["btc_distance_to_beat"] = source["btc_price"] - source["price_to_beat"]
    source["btc_distance_to_beat_pct"] = source["btc_distance_to_beat"] / source["price_to_beat"]
    source["btc_above_price_to_beat"] = (source["btc_price"] > source["price_to_beat"]).astype(int)
    source["direction_signal"] = source["btc_above_price_to_beat"]
    source["market_confidence_gap"] = source["direction_signal"] - source["up_price"]
    source["abs_market_confidence_gap"] = source["market_confidence_gap"].abs()
    source["up_position_value"] = source["up_tokens_before"] * source["up_price"]
    source["down_position_value"] = source["down_tokens_before"] * source["down_price"]
    source["position_value"] = source["up_position_value"] + source["down_position_value"]
    source["position_exposure_pct"] = source["position_value"] / source["balance_before"]
    source["net_position_tokens"] = source["up_tokens_before"] - source["down_tokens_before"]
    source["net_position_value"] = source["up_position_value"] - source["down_position_value"]
    source["target_reward_to_go"] = source["final_balance"] - source["balance_before"]
    for action_column in ACTION_COLUMNS:
        source[action_column] = 0
    source.loc[source["action"] == "hold", "action_hold"] = 1
    source.loc[source["action"] == "buy_up", "action_buy_up"] = 1
    source.loc[source["action"] == "buy_down", "action_buy_down"] = 1

    output = source.copy()
    output["market_id"] = market_id
    output["trajectory_file"] = str(csv_path)
    output["strategy_run_name"] = strategy_run_name

    for column in OUTPUT_COLUMNS:
        if column not in output.columns:
            output[column] = ""

    return output[OUTPUT_COLUMNS], ""


def clean_output_frame(df):
    numeric_columns = [column for column in FEATURE_COLUMNS + ["final_balance", "total_reward", "target_reward_to_go"] if column in df.columns]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace([math.inf, -math.inf], pd.NA)
    df = df.dropna(subset=FEATURE_COLUMNS + ["target_reward_to_go"])
    return df[OUTPUT_COLUMNS]


def build_training_dataset(trajectories_folder, output_path):
    trajectories_folder = Path(trajectories_folder)
    output_path = Path(output_path)
    csv_files = sorted(trajectories_folder.rglob("*.csv"))

    print(f"Trajectories folder: {trajectories_folder}")
    print(f"Output path:         {output_path}")
    print(f"Files found:         {len(csv_files)}")

    processed_files = 0
    skipped = []
    output_rows = 0

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temp_output_path = output_path.with_name(f"{output_path.name}.tmp")
    if temp_output_path.exists():
        temp_output_path.unlink()

    for csv_path in csv_files:
        try:
            file_frame, reason = build_rows_for_file(csv_path, trajectories_folder)
            if reason:
                skipped.append((csv_path, reason))
                continue
            file_frame = clean_output_frame(file_frame)
            file_frame.to_csv(
                temp_output_path,
                mode="a",
                header=not temp_output_path.exists(),
                index=False,
            )
            output_rows += len(file_frame)
            processed_files += 1
            if processed_files % 50 == 0:
                print(f"Processed {processed_files}/{len(csv_files)} files, rows so far: {output_rows}")
        except Exception as e:
            skipped.append((csv_path, f"{type(e).__name__}: {e}"))

    if not temp_output_path.exists():
        pd.DataFrame(columns=OUTPUT_COLUMNS).to_csv(temp_output_path, index=False)

    temp_output_path.replace(output_path)

    print()
    print("Done.")
    print(f"Files processed:     {processed_files}")
    print(f"Files skipped:       {len(skipped)}")
    print(f"Output rows:         {output_rows}")
    print(f"Written:             {output_path}")

    if skipped:
        print()
        print("Skipped files:")
        for csv_path, reason in skipped:
            print(f"- {csv_path}: {reason}")


def main():
    args = parse_args()
    trajectories_folder = resolve_path(
        args.trajectories_folder,
        "Trajectories folder",
        DEFAULT_TRAJECTORIES_FOLDER,
    )
    output_path = resolve_path(args.output, "Output path", DEFAULT_OUTPUT_PATH)
    build_training_dataset(trajectories_folder, output_path)


if __name__ == "__main__":
    main()
