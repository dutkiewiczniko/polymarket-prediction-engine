import argparse
import json
import math
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler


DEFAULT_DATASET_PATH = Path("datasets/training_dataset_v1.csv")
DEFAULT_OUTPUT_DIR = Path("models/reward_model_v1")
DEFAULT_RANDOM_SEED = 42

TARGET_COLUMN = "target_reward_to_go"
MARKET_ID_COLUMN = "market_id"

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

LOAD_COLUMNS = [MARKET_ID_COLUMN] + FEATURE_COLUMNS + [TARGET_COLUMN]


def parse_args():
    parser = argparse.ArgumentParser(description="Train a reward-to-go regression model.")
    parser.add_argument("--dataset", help=f"Training dataset CSV. Default: {DEFAULT_DATASET_PATH}")
    parser.add_argument("--output-dir", help=f"Directory to save model artifacts. Default: {DEFAULT_OUTPUT_DIR}")
    parser.add_argument("--seed", type=int, default=DEFAULT_RANDOM_SEED, help=f"Random seed. Default: {DEFAULT_RANDOM_SEED}")
    parser.add_argument("--batch-size", type=int, default=1024, help="Keras training batch size. Default: 1024")
    parser.add_argument("--epochs", type=int, default=100, help="Maximum Keras training epochs. Default: 100")
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=25,
        help="Epochs without val_loss improvement before early stopping. Default: 25",
    )
    parser.add_argument(
        "--min-epochs-before-stopping",
        type=int,
        default=25,
        help="Do not allow early stopping before this epoch. Default: 25",
    )
    parser.add_argument(
        "--reduce-lr-patience",
        type=int,
        default=8,
        help="Epochs without val_loss improvement before reducing learning rate. Default: 8",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Train for all requested epochs, while still reducing learning rate on plateaus.",
    )
    return parser.parse_args()


def prompt_path(cli_value, prompt, default):
    if cli_value:
        return Path(cli_value)
    value = input(f"{prompt} [{default}]: ").strip()
    return Path(value) if value else default


def load_dataset(dataset_path: Path) -> pd.DataFrame:
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")

    header = pd.read_csv(dataset_path, nrows=0)
    missing = [column for column in LOAD_COLUMNS if column not in header.columns]
    if missing:
        raise ValueError(f"Dataset is missing required columns: {missing}")

    df = pd.read_csv(dataset_path, usecols=LOAD_COLUMNS)
    before_rows = len(df)

    for column in FEATURE_COLUMNS + [TARGET_COLUMN]:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.replace([math.inf, -math.inf], np.nan)
    df = df.dropna(subset=FEATURE_COLUMNS + [TARGET_COLUMN, MARKET_ID_COLUMN])
    df[MARKET_ID_COLUMN] = df[MARKET_ID_COLUMN].astype(str)

    print(f"Rows loaded:          {before_rows}")
    print(f"Rows after cleaning:  {len(df)}")
    print(f"Unique markets:       {df[MARKET_ID_COLUMN].nunique()}")
    return df


def split_market_ids(market_ids, seed):
    market_ids = np.array(sorted(set(market_ids)), dtype=object)
    if len(market_ids) < 3:
        raise ValueError("Need at least 3 unique market_id values for train/validation/test split.")

    rng = np.random.default_rng(seed)
    rng.shuffle(market_ids)

    train_end = max(1, int(len(market_ids) * 0.70))
    val_count = max(1, int(len(market_ids) * 0.15))
    test_count = len(market_ids) - train_end - val_count

    if test_count < 1:
        test_count = 1
        if val_count > 1:
            val_count -= 1
        else:
            train_end -= 1

    val_end = train_end + val_count
    train_ids = market_ids[:train_end]
    val_ids = market_ids[train_end:val_end]
    test_ids = market_ids[val_end:]

    return {
        "train": train_ids.tolist(),
        "validation": val_ids.tolist(),
        "test": test_ids.tolist(),
    }


def split_dataframe(df, split_ids):
    train = df[df[MARKET_ID_COLUMN].isin(split_ids["train"])].copy()
    validation = df[df[MARKET_ID_COLUMN].isin(split_ids["validation"])].copy()
    test = df[df[MARKET_ID_COLUMN].isin(split_ids["test"])].copy()

    print()
    print("Split by market_id")
    print(f"Train markets:       {len(split_ids['train'])}")
    print(f"Validation markets:  {len(split_ids['validation'])}")
    print(f"Test markets:        {len(split_ids['test'])}")
    print(f"Train rows:          {len(train)}")
    print(f"Validation rows:     {len(validation)}")
    print(f"Test rows:           {len(test)}")

    if train.empty or validation.empty or test.empty:
        raise ValueError("Train, validation, and test splits must all contain at least one row.")

    return train, validation, test


def target_distribution(y):
    y = np.asarray(y, dtype=float).reshape(-1)
    return {
        "mean": float(np.mean(y)),
        "std": float(np.std(y)),
        "min": float(np.min(y)),
        "p05": float(np.percentile(y, 5)),
        "p25": float(np.percentile(y, 25)),
        "median": float(np.percentile(y, 50)),
        "p75": float(np.percentile(y, 75)),
        "p95": float(np.percentile(y, 95)),
        "max": float(np.max(y)),
    }


def safe_correlation(y_true, y_pred):
    if len(y_true) < 2 or np.std(y_true) == 0 or np.std(y_pred) == 0:
        return 0.0
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def sign_metrics(y_true, y_pred):
    actual_positive = y_true > 0
    pred_positive = y_pred > 0

    true_positive = np.sum(actual_positive & pred_positive)
    false_positive = np.sum(~actual_positive & pred_positive)
    true_negative = np.sum(~actual_positive & ~pred_positive)
    false_negative = np.sum(actual_positive & ~pred_positive)

    positive_precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    positive_recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    negative_precision = true_negative / (true_negative + false_negative) if (true_negative + false_negative) else 0.0
    negative_recall = true_negative / (true_negative + false_positive) if (true_negative + false_positive) else 0.0

    return {
        "directional_accuracy": float(np.mean(actual_positive == pred_positive)),
        "actual_positive_rate": float(np.mean(actual_positive)),
        "predicted_positive_rate": float(np.mean(pred_positive)),
        "positive_precision": float(positive_precision),
        "positive_recall": float(positive_recall),
        "negative_precision": float(negative_precision),
        "negative_recall": float(negative_recall),
        "true_positive": int(true_positive),
        "false_positive": int(false_positive),
        "true_negative": int(true_negative),
        "false_negative": int(false_negative),
    }


def prediction_lift_metrics(y_true, y_pred):
    order = np.argsort(y_pred)
    n = len(y_true)
    if n == 0:
        return {}

    result = {}
    for label, fraction in [("top_1pct", 0.01), ("top_5pct", 0.05), ("top_10pct", 0.10), ("bottom_10pct", 0.10)]:
        count = max(1, int(n * fraction))
        indexes = order[:count] if label.startswith("bottom") else order[-count:]
        actual = y_true[indexes]
        predicted = y_pred[indexes]
        result[label] = {
            "rows": int(count),
            "mean_actual_target": float(np.mean(actual)),
            "mean_prediction": float(np.mean(predicted)),
            "actual_positive_rate": float(np.mean(actual > 0)),
        }
    return result


def evaluate_predictions(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    error = y_pred - y_true
    abs_error = np.abs(error)
    signs = sign_metrics(y_true, y_pred)

    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "mean_error_bias": float(np.mean(error)),
        "median_absolute_error": float(np.median(abs_error)),
        "p90_absolute_error": float(np.percentile(abs_error, 90)),
        "p95_absolute_error": float(np.percentile(abs_error, 95)),
        "max_absolute_error": float(np.max(abs_error)),
        "correlation": safe_correlation(y_true, y_pred),
        **signs,
        "target": target_distribution(y_true),
        "prediction": target_distribution(y_pred),
        "error": target_distribution(error),
        "lift": prediction_lift_metrics(y_true, y_pred),
    }


def build_model(n_features):
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError(
            "TensorFlow is not installed. Install requirements first: pip install -r requirements.txt"
        ) from e

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(n_features,)),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.15),
        tf.keras.layers.Dense(64, activation="relu"),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.10),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(1),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=tf.keras.losses.Huber(),
    )
    return model, tf


def save_json(path, value):
    with path.open("w", encoding="utf-8") as f:
        json.dump(value, f, indent=2)


def print_metric_block(label, metrics):
    lift_top_10 = metrics["lift"]["top_10pct"]
    print(
        f"{label:<8} MAE={metrics['mae']:.6f} "
        f"RMSE={metrics['rmse']:.6f} "
        f"MedAE={metrics['median_absolute_error']:.6f} "
        f"P90AE={metrics['p90_absolute_error']:.6f} "
        f"R2={metrics['r2']:.6f}"
    )
    print(
        f"{'':<8} Bias={metrics['mean_error_bias']:.6f} "
        f"Corr={metrics['correlation']:.4f} "
        f"DirAcc={metrics['directional_accuracy']:.2%} "
        f"Pred+={metrics['predicted_positive_rate']:.2%} "
        f"Actual+={metrics['actual_positive_rate']:.2%}"
    )
    print(
        f"{'':<8} PosPrec={metrics['positive_precision']:.2%} "
        f"PosRecall={metrics['positive_recall']:.2%} "
        f"Top10% actual mean={lift_top_10['mean_actual_target']:.6f} "
        f"Top10% win rate={lift_top_10['actual_positive_rate']:.2%}"
    )


def compare_against_baseline(baseline_metrics, model_metrics):
    return {
        "mae_delta": float(model_metrics["mae"] - baseline_metrics["mae"]),
        "mae_improvement_pct": float((baseline_metrics["mae"] - model_metrics["mae"]) / baseline_metrics["mae"])
        if baseline_metrics["mae"] else 0.0,
        "rmse_delta": float(model_metrics["rmse"] - baseline_metrics["rmse"]),
        "rmse_improvement_pct": float((baseline_metrics["rmse"] - model_metrics["rmse"]) / baseline_metrics["rmse"])
        if baseline_metrics["rmse"] else 0.0,
        "directional_accuracy_delta": float(
            model_metrics["directional_accuracy"] - baseline_metrics["directional_accuracy"]
        ),
        "top_10pct_actual_mean_delta": float(
            model_metrics["lift"]["top_10pct"]["mean_actual_target"]
            - baseline_metrics["lift"]["top_10pct"]["mean_actual_target"]
        ),
    }


def main():
    args = parse_args()
    dataset_path = prompt_path(args.dataset, "Dataset path", DEFAULT_DATASET_PATH)
    output_dir = prompt_path(args.output_dir, "Output model directory", DEFAULT_OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = load_dataset(dataset_path)
    split_ids = split_market_ids(df[MARKET_ID_COLUMN].unique(), args.seed)
    train_df, val_df, test_df = split_dataframe(df, split_ids)

    x_train = train_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_train = train_df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    x_val = val_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_val = val_df[TARGET_COLUMN].to_numpy(dtype=np.float32)
    x_test = test_df[FEATURE_COLUMNS].to_numpy(dtype=np.float32)
    y_test = test_df[TARGET_COLUMN].to_numpy(dtype=np.float32)

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    baseline_mean = float(np.mean(y_train))
    baseline_val_pred = np.full_like(y_val, baseline_mean, dtype=np.float32)
    baseline_test_pred = np.full_like(y_test, baseline_mean, dtype=np.float32)

    model, tf = build_model(len(FEATURE_COLUMNS))
    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            patience=args.reduce_lr_patience,
            factor=0.5,
            min_lr=1e-6,
        ),
    ]
    if not args.no_early_stopping:
        callbacks.append(
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=args.early_stopping_patience,
                restore_best_weights=True,
                start_from_epoch=min(args.min_epochs_before_stopping, max(args.epochs - 1, 0)),
            )
        )

    history = model.fit(
        x_train_scaled,
        y_train,
        validation_data=(x_val_scaled, y_val),
        batch_size=args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks,
        verbose=1,
    )

    val_pred = model.predict(x_val_scaled, batch_size=args.batch_size).reshape(-1)
    test_pred = model.predict(x_test_scaled, batch_size=args.batch_size).reshape(-1)

    metrics = {
        "dataset_path": str(dataset_path),
        "output_dir": str(output_dir),
        "seed": args.seed,
        "feature_count": len(FEATURE_COLUMNS),
        "training_config": {
            "batch_size": args.batch_size,
            "max_epochs": args.epochs,
            "early_stopping_enabled": not args.no_early_stopping,
            "early_stopping_patience": args.early_stopping_patience,
            "min_epochs_before_stopping": args.min_epochs_before_stopping,
            "reduce_lr_patience": args.reduce_lr_patience,
        },
        "row_counts": {
            "train": int(len(train_df)),
            "validation": int(len(val_df)),
            "test": int(len(test_df)),
        },
        "market_counts": {
            "train": int(len(split_ids["train"])),
            "validation": int(len(split_ids["validation"])),
            "test": int(len(split_ids["test"])),
        },
        "baseline_mean_prediction": baseline_mean,
        "baseline": {
            "validation": evaluate_predictions(y_val, baseline_val_pred),
            "test": evaluate_predictions(y_test, baseline_test_pred),
        },
        "tensorflow_mlp": {
            "validation": evaluate_predictions(y_val, val_pred),
            "test": evaluate_predictions(y_test, test_pred),
            "epochs_trained": int(len(history.history.get("loss", []))),
            "best_val_loss": float(np.min(history.history.get("val_loss", [math.nan]))),
            "final_train_loss": float(history.history.get("loss", [math.nan])[-1]),
            "final_val_loss": float(history.history.get("val_loss", [math.nan])[-1]),
        },
    }
    metrics["comparison_vs_baseline"] = {
        "validation": compare_against_baseline(
            metrics["baseline"]["validation"],
            metrics["tensorflow_mlp"]["validation"],
        ),
        "test": compare_against_baseline(
            metrics["baseline"]["test"],
            metrics["tensorflow_mlp"]["test"],
        ),
    }

    print()
    print("Validation metrics")
    print("------------------")
    print_metric_block("Baseline", metrics["baseline"]["validation"])
    print_metric_block("MLP", metrics["tensorflow_mlp"]["validation"])

    print()
    print("Test metrics")
    print("------------")
    print_metric_block("Baseline", metrics["baseline"]["test"])
    print_metric_block("MLP", metrics["tensorflow_mlp"]["test"])

    print()
    print("Training summary")
    print("----------------")
    print(f"Epochs trained:       {metrics['tensorflow_mlp']['epochs_trained']}")
    print(f"Best val loss:        {metrics['tensorflow_mlp']['best_val_loss']:.6f}")
    print(f"Final train loss:     {metrics['tensorflow_mlp']['final_train_loss']:.6f}")
    print(f"Final val loss:       {metrics['tensorflow_mlp']['final_val_loss']:.6f}")
    print(f"Early stopping:       {'off' if args.no_early_stopping else 'on'}")
    if not args.no_early_stopping:
        print(f"Early stop patience:  {args.early_stopping_patience}")
        print(f"Min stop epoch:       {args.min_epochs_before_stopping}")

    print()
    print("MLP vs baseline")
    print("---------------")
    for split_name in ["validation", "test"]:
        comparison = metrics["comparison_vs_baseline"][split_name]
        print(
            f"{split_name:<10} MAE improvement={comparison['mae_improvement_pct']:.2%} "
            f"RMSE improvement={comparison['rmse_improvement_pct']:.2%} "
            f"DirAcc delta={comparison['directional_accuracy_delta']:.2%} "
            f"Top10 actual mean delta={comparison['top_10pct_actual_mean_delta']:.6f}"
        )

    model.save(output_dir / "model.keras")
    joblib.dump(scaler, output_dir / "scaler.joblib")
    save_json(output_dir / "feature_columns.json", FEATURE_COLUMNS)
    save_json(output_dir / "metrics.json", metrics)
    save_json(output_dir / "split_markets.json", split_ids)

    sample_size = min(10_000, len(test_df))
    predictions_sample = pd.DataFrame({
        MARKET_ID_COLUMN: test_df[MARKET_ID_COLUMN].iloc[:sample_size].to_numpy(),
        "y_true": y_test[:sample_size],
        "y_pred": test_pred[:sample_size],
        "error": test_pred[:sample_size] - y_test[:sample_size],
    })
    predictions_sample.to_csv(output_dir / "predictions_test_sample.csv", index=False)

    print()
    print(f"Saved model artifacts to: {output_dir}")


if __name__ == "__main__":
    main()
