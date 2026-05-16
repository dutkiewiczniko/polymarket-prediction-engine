from pathlib import Path
import argparse
import random

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
        "balance_before",
        "balance_after",
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Trajectory CSV is missing required columns: {sorted(missing)}")
    if "action" in df.columns:
        df["plot_action"] = df["action"].fillna("hold").astype(str)
    elif "action_executed" in df.columns:
        df["plot_action"] = df["action_executed"].fillna("hold").astype(str)
    else:
        raise ValueError("Trajectory CSV is missing an action column: expected 'action' or 'action_executed'")

    # Convert important columns to numeric.
    numeric_columns = [
        "elapsed",
        "up_price",
        "down_price",
        "balance_before",
        "balance_after",
        "final_balance",
        "total_reward",
    ]
    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["elapsed", "up_price", "down_price"])
    df = df.sort_values("elapsed").reset_index(drop=True)

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
        action_rows = df[df["plot_action"] == action]

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
    plt.close(fig)


def plot_balance(df: pd.DataFrame, title: str):
    """Plot simulated balance over time."""

    fig, ax = plt.subplots(figsize=(13, 5))

    ax.plot(df["elapsed"], df["balance_after"], label="Balance after decision")

    trade_rows = df[df["plot_action"] != "hold"]
    if not trade_rows.empty:
        ax.scatter(
            trade_rows["elapsed"],
            trade_rows["balance_after"],
            marker="o",
            s=60,
            label="Trade decision",
        )

    if "final_balance" in df.columns:
        final_values = df["final_balance"].dropna()
        if not final_values.empty:
            final_balance = final_values.iloc[-1]
            final_elapsed = df["elapsed"].dropna().max()
            ax.axhline(
                final_balance,
                linestyle="--",
                linewidth=1.5,
                color="black",
                alpha=0.65,
                label=f"Final resolved balance {final_balance:.2f}",
            )
            ax.scatter(
                [final_elapsed],
                [final_balance],
                marker="*",
                s=140,
                color="black",
                label="Market resolution",
            )

    ax.set_title(title)
    ax.set_xlabel("Elapsed time in market (seconds)")
    ax.set_ylabel("Simulated account value / resolved payout")
    ax.grid(True, alpha=0.3)
    ax.legend()

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def print_trade_summary(df: pd.DataFrame):
    trade_rows = df[df["plot_action"] != "hold"]

    print()
    print("Trade summary")
    print("-------------")
    if "final_balance" in df.columns and df["final_balance"].notna().any():
        final_balance = df["final_balance"].dropna().iloc[-1]
        final_outcome = df["final_outcome"].dropna().iloc[-1] if "final_outcome" in df.columns and df["final_outcome"].notna().any() else ""
        total_reward = df["total_reward"].dropna().iloc[-1] if "total_reward" in df.columns and df["total_reward"].notna().any() else None
        reward_text = f", total_reward={total_reward:.4f}" if total_reward is not None else ""
        print(f"Final outcome={final_outcome}, final_balance={final_balance:.4f}{reward_text}")

    if trade_rows.empty:
        print("No buy/sell actions found in this simulated run.")
        return

    summary_cols = [
        "timestamp",
        "elapsed",
        "seconds_left",
        "plot_action",
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


def find_trajectory_files(folder: str | Path, recursive=True) -> list[Path]:
    folder = Path(folder)
    if not folder.exists():
        raise FileNotFoundError(f"Could not find trajectory folder: {folder}")
    if not folder.is_dir():
        raise ValueError(f"Expected a folder, got: {folder}")
    pattern = "**/*.csv" if recursive else "*.csv"
    return sorted(path for path in folder.glob(pattern) if path.is_file())


def select_trajectory_paths(args) -> list[Path]:
    input_path = args.trajectory_csv

    if args.trajectory_folder:
        folder = Path(args.trajectory_folder)
    elif input_path:
        maybe_path = Path(input_path)
        folder = maybe_path if maybe_path.is_dir() else None
    else:
        prompt_value = input("Path to trajectory CSV or folder: ").strip()
        maybe_path = Path(prompt_value)
        folder = maybe_path if maybe_path.is_dir() else None
        input_path = prompt_value

    if folder is None:
        return [Path(input_path)]

    files = find_trajectory_files(folder, recursive=not args.no_recursive)
    if not files:
        raise FileNotFoundError(f"No CSV files found in trajectory folder: {folder}")

    sample_size = args.sample if args.sample is not None else min(3, len(files))
    sample_size = max(1, min(sample_size, len(files)))
    rng = random.Random(args.seed)
    return sorted(rng.sample(files, sample_size))


def main():
    parser = argparse.ArgumentParser(description="Plot buy/sell decisions from a simulated trajectory CSV.")
    parser.add_argument(
        "trajectory_csv",
        nargs="?",
        default=DEFAULT_TRAJECTORY,
        help="Path to a trajectory CSV or a folder containing trajectory CSVs.",
    )
    parser.add_argument(
        "--trajectory-folder",
        help="Folder to sample trajectory CSVs from, e.g. runs/third_batch_test/trajectories.",
    )
    parser.add_argument(
        "--sample",
        type=int,
        default=None,
        help="Number of random trajectory CSVs to plot when using a folder. Default: 3.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used for folder sampling.",
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only look at CSV files directly inside the folder.",
    )
    parser.add_argument(
        "--no-balance",
        action="store_true",
        help="Only show the price/action chart, not the balance chart.",
    )

    args = parser.parse_args()
    csv_paths = select_trajectory_paths(args)

    print("Trajectory files selected:")
    for csv_path in csv_paths:
        print(f"  {csv_path}")

    for csv_path in csv_paths:
        df = load_trajectory(csv_path)
        path = Path(csv_path)
        title_base = f"{path.parent.name} / {path.stem}"

        print()
        print(f"=== {csv_path} ===")
        print_trade_summary(df)
        plot_price_with_actions(df, f"Buy/Sell Decisions - {title_base}")

        if not args.no_balance:
            plot_balance(df, f"Simulated Balance - {title_base}")


if __name__ == "__main__":
    main()
