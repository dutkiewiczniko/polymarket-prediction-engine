import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from simulator.execution import execute_action
from simulator.model_loader import load_reward_model
from simulator.portfolio import Portfolio
from simulator.replay import infer_final_outcome, load_market_ticks
from scripts.training.train_reward_model import FEATURE_COLUMNS


DEFAULT_OUTPUT_CSV = Path("runs/ml_inference_test.csv")
DEFAULT_MODEL_PATH = Path("models/reward_model.pkl")
DEFAULT_MINIMUM_EDGE = 0.10
DEFAULT_STARTING_BALANCE = 100.0
DEFAULT_ORDER_USD = 1.0

ACTIONS = ["hold", "buy_up", "buy_down"]
LOOKBACK_WINDOWS = [0.5, 1.0, 5.0, 15.0, 30.0]
BTC_RETURN_WINDOWS = [0.5, 1.0, 5.0, 15.0, 30.0, 60.0]
BTC_VOLATILITY_WINDOWS = [5.0, 15.0, 30.0, 60.0]


def parse_args():
    parser = argparse.ArgumentParser(description="Offline ML reward-model replay on a recorded market CSV.")
    parser.add_argument("--market-csv", help="Recorded market CSV to replay.")
    parser.add_argument("--model", help=f"Model artifact path. Default: {DEFAULT_MODEL_PATH}")
    parser.add_argument("--output", help=f"Output trajectory CSV. Default: {DEFAULT_OUTPUT_CSV}")
    parser.add_argument("--minimum-edge", type=float, help=f"Minimum score edge over hold. Default: {DEFAULT_MINIMUM_EDGE}")
    parser.add_argument("--starting-balance", type=float, help=f"Starting simulated cash. Default: {DEFAULT_STARTING_BALANCE}")
    parser.add_argument("--order-usd", type=float, help=f"USD to spend per buy action. Default: {DEFAULT_ORDER_USD}")
    parser.add_argument("--final-outcome", choices=["up", "down"], default=None, help="Optional final outcome override.")
    parser.add_argument("--log-every", type=int, default=1, help="Print every Nth decision row. Default: 1")
    return parser.parse_args()


def default_market_csv():
    data_dir = Path("data")
    files = sorted(data_dir.glob("btc-updown-5m-*.csv"))
    if files:
        return files[-1]
    return Path("data/btc-updown-5m-example.csv")


def prompt_path(cli_value, prompt, default, interactive):
    if cli_value:
        return Path(cli_value)
    if not interactive:
        return default
    value = input(f"{prompt} [{default}]: ").strip()
    return Path(value) if value else default


def prompt_float(cli_value, prompt, default, interactive):
    if cli_value is not None:
        return float(cli_value)
    if not interactive:
        return float(default)
    value = input(f"{prompt} [{default}]: ").strip()
    return float(value) if value else float(default)


def window_suffix(window):
    return str(window).replace(".", "_").replace("_0", "")


def latest_value_at_or_before(series, cutoff_elapsed):
    for elapsed, value in reversed(series):
        if elapsed <= cutoff_elapsed:
            return value
    return None


def raw_change(series, elapsed, window):
    if not series:
        return 0.0
    current = series[-1][1]
    previous = latest_value_at_or_before(series, elapsed - window)
    if current is None or previous is None:
        return 0.0
    return current - previous


def decimal_return(series, elapsed, window):
    if not series:
        return 0.0
    current = series[-1][1]
    previous = latest_value_at_or_before(series, elapsed - window)
    if current is None or previous in (None, 0):
        return 0.0
    return (current - previous) / previous


def btc_volatility(series, elapsed, window):
    values = [value for ts, value in series if ts >= elapsed - window and ts <= elapsed and value not in (None, 0)]
    if len(values) < 3:
        return 0.0

    returns = []
    for previous, current in zip(values, values[1:]):
        if previous:
            returns.append((current - previous) / previous)

    if len(returns) < 2:
        return 0.0

    return float(np.std(returns))


def build_market_feature_row(tick, history):
    btc_price = tick.btc_chainlink if tick.btc_chainlink is not None else tick.btc_binance
    elapsed = tick.elapsed

    up_series = history["up_price"]
    down_series = history["down_price"]
    btc_series = history["btc_price"]

    row = {
        "elapsed": elapsed,
        "seconds_left": tick.seconds_left,
        "time_fraction_elapsed": elapsed / 300.0 if elapsed is not None else math.nan,
        "up_price": tick.up_price,
        "down_price": tick.down_price,
        "up_minus_down": tick.up_price - tick.down_price,
        "btc_price": btc_price,
        "price_to_beat": tick.price_to_beat,
    }

    for window in LOOKBACK_WINDOWS:
        suffix = window_suffix(window)
        row[f"up_price_change_{suffix}s"] = raw_change(up_series, elapsed, window)
        row[f"down_price_change_{suffix}s"] = raw_change(down_series, elapsed, window)

    for window in BTC_RETURN_WINDOWS:
        suffix = window_suffix(window)
        row[f"btc_return_{suffix}s"] = decimal_return(btc_series, elapsed, window)

    for window in BTC_VOLATILITY_WINDOWS:
        suffix = window_suffix(window)
        row[f"btc_volatility_{suffix}s"] = btc_volatility(btc_series, elapsed, window)

    row["btc_distance_to_beat"] = btc_price - tick.price_to_beat
    row["btc_distance_to_beat_pct"] = row["btc_distance_to_beat"] / tick.price_to_beat if tick.price_to_beat else math.nan
    row["btc_above_price_to_beat"] = 1 if btc_price > tick.price_to_beat else 0
    row["direction_signal"] = row["btc_above_price_to_beat"]
    row["market_confidence_gap"] = row["direction_signal"] - tick.up_price
    row["abs_market_confidence_gap"] = abs(row["market_confidence_gap"])
    return row


def portfolio_feature_row(portfolio, up_price, down_price):
    balance_before = portfolio.mark_to_market(up_price, down_price)
    up_position_value = portfolio.up_tokens * up_price
    down_position_value = portfolio.down_tokens * down_price
    position_value = up_position_value + down_position_value

    return {
        "cash_before": portfolio.cash,
        "up_tokens_before": portfolio.up_tokens,
        "down_tokens_before": portfolio.down_tokens,
        "balance_before": balance_before,
        "up_position_value": up_position_value,
        "down_position_value": down_position_value,
        "position_value": position_value,
        "position_exposure_pct": position_value / balance_before if balance_before else 0.0,
        "net_position_tokens": portfolio.up_tokens - portfolio.down_tokens,
        "net_position_value": up_position_value - down_position_value,
    }


def build_prediction_rows(market_row, portfolio_state) -> pd.DataFrame:
    rows = []
    for action in ACTIONS:
        row = {**market_row, **portfolio_state}
        row["action_hold"] = 1 if action == "hold" else 0
        row["action_buy_up"] = 1 if action == "buy_up" else 0
        row["action_buy_down"] = 1 if action == "buy_down" else 0
        row["candidate_action"] = action
        rows.append(row)

    return pd.DataFrame(rows)


def validate_prediction_frame(candidates, feature_columns):
    missing = [column for column in feature_columns if column not in candidates.columns]
    if missing:
        raise ValueError(f"Prediction rows are missing model features: {missing}")

    features = candidates[feature_columns].apply(pd.to_numeric, errors="coerce")
    features = features.replace([math.inf, -math.inf], np.nan)
    if features.isna().any().any():
        bad_columns = features.columns[features.isna().any()].tolist()
        raise ValueError(f"Prediction rows contain invalid feature values in: {bad_columns}")

    return features


def update_history(history, tick):
    btc_price = tick.btc_chainlink if tick.btc_chainlink is not None else tick.btc_binance
    history["up_price"].append((tick.elapsed, tick.up_price))
    history["down_price"].append((tick.elapsed, tick.down_price))
    history["btc_price"].append((tick.elapsed, btc_price))


def replay_with_model(
    *,
    market_csv,
    model_path,
    output_csv,
    minimum_edge,
    starting_balance,
    order_usd,
    final_outcome=None,
    log_every=1,
):
    ticks = load_market_ticks(market_csv)
    ticks = [
        tick for tick in ticks
        if tick.elapsed is not None
        and tick.seconds_left is not None
        and tick.up_price is not None
        and tick.down_price is not None
        and tick.price_to_beat is not None
        and (tick.btc_chainlink is not None or tick.btc_binance is not None)
    ]
    if not ticks:
        raise ValueError(f"No replayable ticks with complete ML features found in {market_csv}")

    bundle = load_reward_model(model_path)
    feature_columns = bundle.feature_columns or FEATURE_COLUMNS
    portfolio = Portfolio(cash=starting_balance)
    history = {"up_price": [], "down_price": [], "btc_price": []}
    rows = []
    action_counts = {"hold": 0, "buy_up": 0, "buy_down": 0}

    print(f"Loaded model: {bundle.source_path}")
    print(f"Feature columns: {len(feature_columns)}")
    print(f"Replay ticks: {len(ticks)}")

    for row_no, tick in enumerate(ticks, start=1):
        update_history(history, tick)
        market_row = build_market_feature_row(tick, history)
        portfolio_before = portfolio_feature_row(portfolio, tick.up_price, tick.down_price)
        candidates = build_prediction_rows(market_row, portfolio_before)
        features = validate_prediction_frame(candidates, feature_columns)
        predictions = bundle.predict(features)
        scores = dict(zip(ACTIONS, predictions))

        hold_score = scores["hold"]
        best_action = max(scores, key=scores.get)
        best_score = scores[best_action]

        if best_action != "hold" and best_score > hold_score + minimum_edge:
            chosen_action = best_action
            reason = f"ml edge {best_score - hold_score:.4f} > {minimum_edge:.4f}"
        else:
            chosen_action = "hold"
            reason = f"edge below threshold {best_score - hold_score:.4f} <= {minimum_edge:.4f}"

        events = execute_action(
            portfolio=portfolio,
            action=chosen_action,
            timestamp=tick.timestamp,
            up_price=tick.up_price,
            down_price=tick.down_price,
            usd_amount=order_usd,
            reason=reason,
        )
        action_executed = chosen_action if chosen_action == "hold" or events else "hold"
        if action_executed != chosen_action:
            reason = f"{reason}; no fill"
        action_counts[action_executed] += 1
        balance_after = portfolio.mark_to_market(tick.up_price, tick.down_price)

        if log_every and row_no % log_every == 0:
            print(
                f"elapsed={tick.elapsed:.1f} "
                f"hold={scores['hold']:.4f} "
                f"buy_up={scores['buy_up']:.4f} "
                f"buy_down={scores['buy_down']:.4f} "
                f"-> {action_executed} edge={best_score - hold_score:.4f}"
            )

        rows.append({
            "timestamp": tick.timestamp,
            "unix_time": tick.unix_time,
            "elapsed": tick.elapsed,
            "seconds_left": tick.seconds_left,
            "up_price": tick.up_price,
            "down_price": tick.down_price,
            "btc_price": market_row["btc_price"],
            "price_to_beat": tick.price_to_beat,
            "cash_before": portfolio_before["cash_before"],
            "up_tokens_before": portfolio_before["up_tokens_before"],
            "down_tokens_before": portfolio_before["down_tokens_before"],
            "balance_before": portfolio_before["balance_before"],
            "cash_after": portfolio.cash,
            "up_tokens_after": portfolio.up_tokens,
            "down_tokens_after": portfolio.down_tokens,
            "balance_after": balance_after,
            "predicted_reward_hold": scores["hold"],
            "predicted_reward_buy_up": scores["buy_up"],
            "predicted_reward_buy_down": scores["buy_down"],
            "chosen_action": chosen_action,
            "chosen_score": scores[chosen_action],
            "hold_score": hold_score,
            "score_edge": best_score - hold_score,
            "action_executed": action_executed,
            "reason": reason,
        })

    resolved_outcome = infer_final_outcome(ticks, fallback=final_outcome)
    final_balance = portfolio.resolve(resolved_outcome)
    total_reward = final_balance - starting_balance

    for row in rows:
        row["final_outcome"] = resolved_outcome
        row["final_balance"] = final_balance
        row["total_reward"] = total_reward
        row["reward_to_go"] = final_balance - float(row["balance_before"])

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Replay complete.")
    print(f"final_balance:    {final_balance:.4f}")
    print(f"total_reward:     {total_reward:.4f}")
    print(f"buy_up actions:   {action_counts['buy_up']}")
    print(f"buy_down actions: {action_counts['buy_down']}")
    print(f"holds:            {action_counts['hold']}")
    print(f"output_csv:       {output_csv}")


def main():
    args = parse_args()
    interactive = len(sys.argv) == 1
    market_csv = prompt_path(args.market_csv, "Market CSV path", default_market_csv(), interactive)
    model_path = prompt_path(args.model, "Model path", DEFAULT_MODEL_PATH, interactive)
    output_csv = prompt_path(args.output, "Output trajectory CSV path", DEFAULT_OUTPUT_CSV, interactive)
    minimum_edge = prompt_float(args.minimum_edge, "minimum_edge", DEFAULT_MINIMUM_EDGE, interactive)
    starting_balance = prompt_float(args.starting_balance, "starting_balance", DEFAULT_STARTING_BALANCE, interactive)
    order_usd = prompt_float(args.order_usd, "order_usd", DEFAULT_ORDER_USD, interactive)

    replay_with_model(
        market_csv=market_csv,
        model_path=model_path,
        output_csv=output_csv,
        minimum_edge=minimum_edge,
        starting_balance=starting_balance,
        order_usd=order_usd,
        final_outcome=args.final_outcome,
        log_every=args.log_every,
    )


if __name__ == "__main__":
    main()
