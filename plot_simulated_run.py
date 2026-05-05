from pathlib import Path
import argparse

import pandas as pd
import matplotlib.pyplot as plt


DEFAULT_TRAJECTORY = ""


def load_trajectory(csv_path: str | Path) -> pd.DataFrame:
    csv_path = Path(csv_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find trajectory CSV: {csv_path}")

    df = pd.read_csv(csv_path)

    required_columns = {
        "elapsed",
        "up_price",
        "down_price",
        "action",
        "balance_before",
        "balance_after",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Trajectory CSV is missing required columns: {sorted(missing)}")

    # Convert important columns to numeric.
    for col in ["elapsed", "up_price", "down_price", "balance_before", "balance_after"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["elapsed", "up_price", "down_price"])

    return df


def plot_price_with_actions(df: pd.DataFrame, title: str):
    """Plot UP/DOWN prices and mark buy/sell actions."""

    fig, ax = plt.subplots(figsize=(13, 7))

    ax.plot(df["elapsed"], df["up_price"], label="UP price")
    ax.plot(df["elapsed"], df["down_price"], label="DOWN price")

    action_styles = {
        "buy_up": {"column": "up_price", "marker": "^", "label": "Buy UP"},
        "buy_down": {"column": "down_price", "marker": "v", "label": "Buy DOWN"},
        "sell_up": {"column": "up_price", "marker": "x", "label": "Sell UP"},
        "sell_down": {"column": "down_price", "marker": "x", "label": "Sell DOWN"},
    }

    for action, style in action_styles.items():
        action_rows = df[df["action"] == action]

        if action_rows.empty:
            continue

        ax.scatter(
            action_rows["elapsed"],
            action_rows[style["column"]],
            marker=style["marker"],
            s=90,
            label=style["label"],
        )

    ax.set_title(title)
    ax.set_xlabel("Elapsed time in market (seconds)")
    ax.set_ylabel("Polymarket price / probability")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_balance(df: pd.DataFrame, title: str):
    """Plot simulated balance over time."""

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(df["elapsed"], df["balance_after"], label="Balance after decision")

    trade_rows = df[df["action"] != "hold"]
    if not trade_rows.empty:
        ax.scatter(
            trade_rows["elapsed"],
            trade_rows["balance_after"],
            marker="o",
            s=60,
            label="Trade decision",
        )

    ax.set_title(title)
    ax.set_xlabel("Elapsed time in market (seconds)")
    ax.set_ylabel("Simulated account value")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()


def print_trade_summary(df: pd.DataFrame):
    trade_rows = df[df["action"] != "hold"]

    print()
    print("Trade summary")
    print("-------------")

    if trade_rows.empty:
        print("No buy/sell actions found in this simulated run.")
        return

    summary_cols = [
        "timestamp",
        "elapsed",
        "seconds_left",
        "action",
        "reason",
        "up_price",
        "down_price",
        "cash_after",
        "up_tokens_after",
        "down_tokens_after",
        "balance_after",
    ]

    available_cols = [col for col in summary_cols if col in trade_rows.columns]
    print(trade_rows[available_cols].to_string(index=False))


def main():
    parser = argparse.ArgumentParser(description="Plot buy/sell decisions from a simulated trajectory CSV.")
    parser.add_argument(
        "trajectory_csv",
        nargs="?",
        default=DEFAULT_TRAJECTORY,
        help="Path to a simulated trajectory CSV, e.g. runs/first_batch_test/trajectories/file.csv",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Only show the price/action chart, not the balance chart.",
    )

    args = parser.parse_args()

    csv_path = args.trajectory_csv

    if not csv_path:
        csv_path = input("Path to trajectory CSV: ").strip()

    df = load_trajectory(csv_path)

    title_base = Path(csv_path).stem

    print_trade_summary(df)
    plot_price_with_actions(df, f"Buy/Sell Decisions - {title_base}")

    if not args.no_balance:
        plot_balance(df, f"Simulated Balance - {title_base}")


if __name__ == "__main__":
    main()
