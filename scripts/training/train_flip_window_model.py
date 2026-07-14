"""Train a flip-window comeback classifier.

This model consumes the gitignored ``flip_windows_v1`` dataset produced by
``scripts/analysis/extract_flip_windows.py``. It keeps ``july_live`` untouched
as the final holdout, does grouped cross-validation by market on ``may_bench``,
and trains a tree model on per-window aggregate features.

Default usage:
    python scripts/training/train_flip_window_model.py
"""

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_DATASET_DIR = Path("flip_windows_v1")
DEFAULT_OUTPUT_DIR = Path("models/flip_window_model_v1")
DEFAULT_TARGET = "won_resolution"
DEFAULT_TRAIN_COHORT = "may_bench"
DEFAULT_HOLDOUT_COHORT = "july_live"
DEFAULT_SEED = 42
BOOLEAN_LABELS = {
    "won_resolution",
    "recovered_50",
    "recovered_75",
    "recovered_90",
    "recovered_95",
}
MANIFEST_FEATURE_COLUMNS = [
    "entry_price",
    "entry_seconds_left",
    "entry_deficit_pct",
    "crash_speed_30s",
    "hour_utc",
    "episode_num",
    "has_hist_300s",
    "has_hist_600s",
    "has_hist_900s",
]
EXCLUDED_TICK_FEATURE_COLUMNS = {
    "unix_time",
}
META_COLUMNS = [
    "window_file",
    "market",
    "side",
    "cohort",
    "episode_num",
    "window_end_reason",
]


@dataclass(frozen=True)
class FeatureConfig:
    framing: str
    early_seconds: float
    target: str
    train_cohort: str
    holdout_cohort: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a grouped-CV flip-window classifier with July holdout."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=DEFAULT_DATASET_DIR,
        help=f"Directory containing manifest.csv and windows/. Default: {DEFAULT_DATASET_DIR}",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for model artifacts. Default: {DEFAULT_OUTPUT_DIR}",
    )
    parser.add_argument(
        "--target",
        default=DEFAULT_TARGET,
        choices=sorted(BOOLEAN_LABELS),
        help=f"Boolean manifest label to predict. Default: {DEFAULT_TARGET}",
    )
    parser.add_argument(
        "--framing",
        choices=["entry", "early"],
        default="entry",
        help="entry uses the first tick only; early aggregates ticks up to --early-seconds.",
    )
    parser.add_argument(
        "--early-seconds",
        type=float,
        default=30.0,
        help="Cutoff for --framing early. Ignored for entry framing. Default: 30.",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Grouped CV folds on the training cohort. Default: 5.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help=f"Random seed. Default: {DEFAULT_SEED}",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=250,
        help="Histogram gradient boosting iterations. Default: 250.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.04,
        help="Histogram gradient boosting learning rate. Default: 0.04.",
    )
    parser.add_argument(
        "--max-leaf-nodes",
        type=int,
        default=15,
        help="Histogram gradient boosting max leaf nodes. Default: 15.",
    )
    parser.add_argument(
        "--l2-regularization",
        type=float,
        default=0.1,
        help="Histogram gradient boosting L2 regularization. Default: 0.1.",
    )
    parser.add_argument(
        "--min-samples-leaf",
        type=int,
        default=15,
        help="Histogram gradient boosting min samples per leaf. Default: 15.",
    )
    parser.add_argument(
        "--train-cohort",
        default=DEFAULT_TRAIN_COHORT,
        help=f"Cohort used for training/CV. Default: {DEFAULT_TRAIN_COHORT}",
    )
    parser.add_argument(
        "--holdout-cohort",
        default=DEFAULT_HOLDOUT_COHORT,
        help=f"Cohort reserved for final evaluation. Default: {DEFAULT_HOLDOUT_COHORT}",
    )
    parser.add_argument(
        "--skip-permutation-importance",
        action="store_true",
        help="Skip slower holdout permutation importance report.",
    )
    parser.add_argument(
        "--leakage-warn-auc",
        type=float,
        default=0.80,
        help=(
            "Univariate AUC above which a single feature is flagged as likely "
            "label leakage. Default: 0.80."
        ),
    )
    parser.add_argument(
        "--drop-leaky-features",
        action="store_true",
        help=(
            "Drop every feature the leakage sentinel flags before training. Use "
            "this when a framing/target combination is known to encode the label "
            "(e.g. early-window price aggregates vs a recovery target)."
        ),
    )
    parser.add_argument(
        "--skip-baselines",
        action="store_true",
        help="Skip the prior / logistic baselines used to measure model lift.",
    )
    return parser.parse_args()


def bool_series(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False})


def load_manifest(dataset_dir: Path, target: str) -> pd.DataFrame:
    manifest_path = dataset_dir / "manifest.csv"
    windows_dir = dataset_dir / "windows"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")
    if not windows_dir.exists():
        raise FileNotFoundError(f"Missing windows directory: {windows_dir}")

    manifest = pd.read_csv(manifest_path)
    required = set(META_COLUMNS + MANIFEST_FEATURE_COLUMNS + [target])
    missing = sorted(required.difference(manifest.columns))
    if missing:
        raise ValueError(f"Manifest is missing required columns: {missing}")

    manifest[target] = bool_series(manifest[target])
    if manifest[target].isna().any():
        raise ValueError(f"Target column contains non-boolean values: {target}")

    for column in MANIFEST_FEATURE_COLUMNS:
        if column.startswith("has_hist_"):
            manifest[column] = bool_series(manifest[column]).astype(float)
        else:
            manifest[column] = pd.to_numeric(manifest[column], errors="coerce")
    return manifest


def numeric_tick_columns(csv_path: Path) -> list[str]:
    header = pd.read_csv(csv_path, nrows=20)
    numeric_cols = []
    for column in header.columns:
        if column in EXCLUDED_TICK_FEATURE_COLUMNS:
            continue
        converted = pd.to_numeric(header[column], errors="coerce")
        if converted.notna().any():
            numeric_cols.append(column)
    return numeric_cols


def entry_features(df: pd.DataFrame, numeric_columns: list[str]) -> dict[str, float]:
    first = df.iloc[0]
    return {f"tick_entry__{column}": float(first[column]) for column in numeric_columns}


def aggregate_features(
    df: pd.DataFrame,
    numeric_columns: list[str],
    early_seconds: float,
) -> dict[str, float]:
    if "time_in_window_s" not in df.columns:
        raise ValueError("Window CSV is missing time_in_window_s")
    window = df[df["time_in_window_s"] <= early_seconds].copy()
    if window.empty:
        window = df.iloc[[0]].copy()

    features: dict[str, float] = {}
    for column in numeric_columns:
        values = pd.to_numeric(window[column], errors="coerce")
        first = values.iloc[0]
        last = values.iloc[-1]
        features[f"tick_entry__{column}"] = float(first)
        features[f"tick_last__{column}"] = float(last)
        features[f"tick_min__{column}"] = float(values.min(skipna=True))
        features[f"tick_max__{column}"] = float(values.max(skipna=True))
        features[f"tick_mean__{column}"] = float(values.mean(skipna=True))
        features[f"tick_std__{column}"] = float(values.std(skipna=True))
        features[f"tick_delta__{column}"] = float(last - first)

    features["tick_cutoff_rows"] = float(len(window))
    features["tick_cutoff_elapsed_s"] = float(window["time_in_window_s"].max())
    return features


def build_feature_frame(
    manifest: pd.DataFrame,
    dataset_dir: Path,
    config: FeatureConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    windows_dir = dataset_dir / "windows"
    first_window = windows_dir / str(manifest["window_file"].iloc[0])
    tick_columns = numeric_tick_columns(first_window)

    rows = []
    for record in manifest.to_dict("records"):
        path = windows_dir / str(record["window_file"])
        if not path.exists():
            raise FileNotFoundError(f"Window CSV not found: {path}")
        tick_df = pd.read_csv(path, usecols=tick_columns)
        if tick_df.empty:
            raise ValueError(f"Window CSV has no ticks: {path}")
        tick_df = tick_df.apply(pd.to_numeric, errors="coerce")

        features = {}
        for column in MANIFEST_FEATURE_COLUMNS:
            features[f"manifest__{column}"] = float(record[column])
        features["manifest__side_is_up"] = 1.0 if record["side"] == "up" else 0.0

        if config.framing == "entry":
            features.update(entry_features(tick_df, tick_columns))
        else:
            features.update(aggregate_features(tick_df, tick_columns, config.early_seconds))
        rows.append(features)

    feature_df = pd.DataFrame(rows)
    feature_df = feature_df.replace([math.inf, -math.inf], np.nan)
    metadata = manifest[META_COLUMNS + [config.target]].copy()
    feature_columns = feature_df.columns.tolist()
    return feature_df, metadata, feature_columns


def make_model(args: argparse.Namespace, positive_rate: float) -> HistGradientBoostingClassifier:
    if positive_rate <= 0 or positive_rate >= 1:
        raise ValueError(f"Need both classes in training data; positive rate={positive_rate:.4f}")
    class_weight = {0: 1.0, 1: float((1.0 - positive_rate) / positive_rate)}
    return HistGradientBoostingClassifier(
        loss="log_loss",
        learning_rate=args.learning_rate,
        max_iter=args.max_iter,
        max_leaf_nodes=args.max_leaf_nodes,
        min_samples_leaf=args.min_samples_leaf,
        l2_regularization=args.l2_regularization,
        class_weight=class_weight,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=25,
        random_state=args.seed,
    )


def hgb_factory(args: argparse.Namespace):
    """Return a factory building the tuned tree model for a given fold."""
    return lambda positive_rate: make_model(args, positive_rate)


def prior_baseline_factory():
    """Constant model that always predicts the training base rate.

    This is the floor: any real model must beat its Brier/log-loss, and its
    ROC/PR sit at chance. If the tree cannot clear this by a meaningful margin,
    the features carry no signal.
    """
    return lambda _positive_rate: DummyClassifier(strategy="prior")


def logistic_baseline_factory(args: argparse.Namespace):
    """Linear baseline on all features (median-imputed + standardized).

    A gradient-boosted tree that fails to beat plain logistic regression is not
    finding non-linear structure worth the complexity. NaNs (e.g. missing
    stitched momentum) are median-imputed so the linear model stays comparable.
    """

    def build(_positive_rate: float) -> Pipeline:
        return Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "logit",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        C=1.0,
                        random_state=args.seed,
                    ),
                ),
            ]
        )

    return build


def leakage_sentinel(
    x: pd.DataFrame,
    y: np.ndarray,
    warn_auc: float,
) -> list[dict]:
    """Rank features by univariate separating power against the target.

    A single trailing-only feature should not, on its own, come close to
    perfectly ordering the label. When one does (high |AUC-0.5|), it is almost
    always encoding the outcome — e.g. an early-window price aggregate that has
    already crossed the recovery threshold the label is defined on. Runs on the
    training cohort only; the holdout is never inspected here.
    """
    report = []
    for column in x.columns:
        values = pd.to_numeric(x[column], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(values)
        if mask.sum() < 10:
            continue
        yc = y[mask]
        if len(np.unique(yc)) < 2:
            continue
        auc = roc_auc_score(yc, values[mask])
        report.append(
            {
                "feature": column,
                "auc": float(auc),
                "separation": float(abs(auc - 0.5)),
                "coverage": float(mask.mean()),
                "suspicious": bool(abs(auc - 0.5) + 0.5 >= warn_auc),
            }
        )
    report.sort(key=lambda item: item["separation"], reverse=True)
    return report


def safe_auc(metric_name: str, y_true: np.ndarray, y_prob: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    if metric_name == "roc":
        return float(roc_auc_score(y_true, y_prob))
    if metric_name == "pr":
        return float(average_precision_score(y_true, y_prob))
    raise ValueError(metric_name)


def threshold_for_recall(y_true: np.ndarray, y_prob: np.ndarray, min_recall: float) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    candidates = [
        (float(p), float(t))
        for p, r, t in zip(precision[:-1], recall[:-1], thresholds)
        if r >= min_recall
    ]
    if not candidates:
        return 0.5
    return max(candidates, key=lambda item: item[0])[1]


def evaluate_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = y_prob >= threshold
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    precision_at = {}
    for pct in (0.05, 0.10, 0.20):
        k = max(1, int(math.ceil(len(y_true) * pct)))
        top_idx = np.argsort(y_prob)[-k:]
        precision_at[f"top_{int(pct * 100)}pct"] = {
            "k": int(k),
            "positive_rate": float(np.mean(y_true[top_idx])),
            "mean_score": float(np.mean(y_prob[top_idx])),
        }

    return {
        "rows": int(len(y_true)),
        "positives": int(np.sum(y_true)),
        "positive_rate": float(np.mean(y_true)),
        "roc_auc": safe_auc("roc", y_true, y_prob),
        "pr_auc": safe_auc("pr", y_true, y_prob),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, np.clip(y_prob, 1e-6, 1 - 1e-6), labels=[0, 1])),
        "threshold": float(threshold),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "predicted_positive_rate": float(np.mean(y_pred)),
        "confusion_matrix_tn_fp_fn_tp": [
            int(cm[0, 0]),
            int(cm[0, 1]),
            int(cm[1, 0]),
            int(cm[1, 1]),
        ],
        "precision_at": precision_at,
    }


def grouped_split_indices(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    args: argparse.Namespace,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Deterministic market-grouped folds shared by every estimator we score.

    Stratifies on the label when possible so rare positives are spread across
    folds; falls back to plain GroupKFold if a stratified split is infeasible.
    """
    unique_groups = np.unique(groups)
    fold_count = min(args.cv_folds, len(unique_groups))
    if fold_count < 2:
        raise ValueError("Need at least two markets for grouped CV.")
    splitter = StratifiedGroupKFold(
        n_splits=fold_count,
        shuffle=True,
        random_state=args.seed,
    )
    try:
        return list(splitter.split(x, y, groups))
    except ValueError:
        return list(GroupKFold(n_splits=fold_count).split(x, y, groups))


def grouped_cv(
    x: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    args: argparse.Namespace,
    model_factory,
    split_indices: list[tuple[np.ndarray, np.ndarray]] | None = None,
) -> tuple[pd.DataFrame, list[dict], float]:
    if split_indices is None:
        split_indices = grouped_split_indices(x, y, groups, args)
    folds = []
    oof_prob = np.full(len(y), np.nan, dtype=float)

    for fold_idx, (train_idx, val_idx) in enumerate(split_indices, start=1):
        model = model_factory(float(np.mean(y[train_idx])))
        model.fit(x.iloc[train_idx], y[train_idx])
        val_prob = model.predict_proba(x.iloc[val_idx])[:, 1]
        oof_prob[val_idx] = val_prob
        threshold = threshold_for_recall(y[train_idx], model.predict_proba(x.iloc[train_idx])[:, 1], 0.50)
        metrics = evaluate_predictions(y[val_idx], val_prob, threshold)
        metrics["fold"] = fold_idx
        metrics["train_rows"] = int(len(train_idx))
        metrics["validation_markets"] = int(len(np.unique(groups[val_idx])))
        folds.append(metrics)

    if np.isnan(oof_prob).any():
        raise RuntimeError("Grouped CV did not score every training row.")
    oof_threshold = threshold_for_recall(y, oof_prob, 0.50)
    oof_metrics = evaluate_predictions(y, oof_prob, oof_threshold)
    return pd.DataFrame({"y_true": y, "y_prob": oof_prob, "market": groups}), folds, oof_threshold


def summarize_manifest(manifest: pd.DataFrame, target: str) -> dict:
    by_cohort = {}
    for cohort, group in manifest.groupby("cohort"):
        y = bool_series(group[target]).astype(bool)
        by_cohort[cohort] = {
            "rows": int(len(group)),
            "markets": int(group["market"].nunique()),
            "positives": int(y.sum()),
            "positive_rate": float(y.mean()),
        }
    return {
        "rows": int(len(manifest)),
        "markets": int(manifest["market"].nunique()),
        "target": target,
        "by_cohort": by_cohort,
    }


def write_json(path: Path, payload: dict | list) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def save_predictions(
    path: Path,
    metadata: pd.DataFrame,
    y_prob: np.ndarray,
    threshold: float,
    target: str,
) -> None:
    out = metadata.copy()
    out["y_true"] = bool_series(out[target]).astype(int)
    out["y_prob"] = y_prob
    out["y_pred"] = (y_prob >= threshold).astype(int)
    out.to_csv(path, index=False)


def print_metrics(name: str, metrics: dict) -> None:
    roc = "n/a" if metrics["roc_auc"] is None else f"{metrics['roc_auc']:.4f}"
    pr = "n/a" if metrics["pr_auc"] is None else f"{metrics['pr_auc']:.4f}"
    print(
        f"{name:<10} rows={metrics['rows']} pos={metrics['positives']} "
        f"rate={metrics['positive_rate']:.2%} ROC={roc} PR={pr} "
        f"Brier={metrics['brier']:.4f}"
    )
    print(
        f"{'':<10} threshold={metrics['threshold']:.4f} "
        f"precision={metrics['precision']:.2%} recall={metrics['recall']:.2%} "
        f"pred+={metrics['predicted_positive_rate']:.2%}"
    )


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    config = FeatureConfig(
        framing=args.framing,
        early_seconds=args.early_seconds,
        target=args.target,
        train_cohort=args.train_cohort,
        holdout_cohort=args.holdout_cohort,
    )
    manifest = load_manifest(args.dataset_dir, args.target)
    x, metadata, feature_columns = build_feature_frame(manifest, args.dataset_dir, config)
    y_all = bool_series(metadata[args.target]).astype(int).to_numpy()

    train_mask = metadata["cohort"].eq(args.train_cohort).to_numpy()
    holdout_mask = metadata["cohort"].eq(args.holdout_cohort).to_numpy()
    if not train_mask.any():
        raise ValueError(f"No rows found for train cohort: {args.train_cohort}")
    if not holdout_mask.any():
        raise ValueError(f"No rows found for holdout cohort: {args.holdout_cohort}")

    x_train = x.loc[train_mask].reset_index(drop=True)
    y_train = y_all[train_mask]
    train_meta = metadata.loc[train_mask].reset_index(drop=True)
    groups = train_meta["market"].astype(str).to_numpy()

    x_holdout = x.loc[holdout_mask].reset_index(drop=True)
    y_holdout = y_all[holdout_mask]
    holdout_meta = metadata.loc[holdout_mask].reset_index(drop=True)

    # Leakage sentinel: inspect single-feature separating power on the training
    # cohort BEFORE fitting anything. Catches label-encoding features (e.g. an
    # early-window price aggregate that already crossed a recovery threshold).
    sentinel = leakage_sentinel(x_train, y_train, args.leakage_warn_auc)
    flagged = [item["feature"] for item in sentinel if item["suspicious"]]
    if flagged:
        print()
        print("LEAKAGE WARNING")
        print("---------------")
        print(
            f"{len(flagged)} feature(s) exceed univariate AUC {args.leakage_warn_auc:.2f} "
            f"on the training cohort -- likely encode the target:"
        )
        for item in sentinel[: min(8, len(flagged))]:
            if item["suspicious"]:
                print(f"  {item['feature']:<32} AUC={item['auc']:.3f} cov={item['coverage']:.0%}")
        if args.drop_leaky_features:
            print(f"Dropping {len(flagged)} flagged feature(s) (--drop-leaky-features).")
        else:
            print("Not dropped. Re-run with --drop-leaky-features, or fix the framing/target.")

    if args.drop_leaky_features and flagged:
        keep = [c for c in feature_columns if c not in set(flagged)]
        x_train = x_train[keep]
        x_holdout = x_holdout[keep]
        feature_columns = keep

    split_indices = grouped_split_indices(x_train, y_train, groups, args)
    oof, fold_metrics, threshold = grouped_cv(
        x_train, y_train, groups, args, hgb_factory(args), split_indices
    )
    oof_metrics = evaluate_predictions(y_train, oof["y_prob"].to_numpy(), threshold)

    model = make_model(args, positive_rate=float(np.mean(y_train)))
    model.fit(x_train, y_train)
    holdout_prob = model.predict_proba(x_holdout)[:, 1]
    holdout_metrics = evaluate_predictions(y_holdout, holdout_prob, threshold)

    baselines: dict[str, dict] = {}
    if not args.skip_baselines:
        for name, factory in (
            ("prior", prior_baseline_factory()),
            ("logistic", logistic_baseline_factory(args)),
        ):
            b_oof, _, b_threshold = grouped_cv(
                x_train, y_train, groups, args, factory, split_indices
            )
            b_model = factory(float(np.mean(y_train)))
            b_model.fit(x_train, y_train)
            b_holdout_prob = b_model.predict_proba(x_holdout)[:, 1]
            baselines[name] = {
                "out_of_fold": evaluate_predictions(
                    y_train, b_oof["y_prob"].to_numpy(), b_threshold
                ),
                "holdout": evaluate_predictions(y_holdout, b_holdout_prob, b_threshold),
            }

    metrics = {
        "dataset": summarize_manifest(manifest, args.target),
        "config": asdict(config),
        "model": {
            "type": "HistGradientBoostingClassifier",
            "max_iter": args.max_iter,
            "learning_rate": args.learning_rate,
            "max_leaf_nodes": args.max_leaf_nodes,
            "l2_regularization": args.l2_regularization,
            "min_samples_leaf": args.min_samples_leaf,
            "random_seed": args.seed,
            "features": len(feature_columns),
            "trained_iterations": int(model.n_iter_),
        },
        "grouped_cv": {
            "folds": fold_metrics,
            "out_of_fold": oof_metrics,
        },
        "holdout": holdout_metrics,
        "baselines": baselines,
        "leakage_sentinel": sentinel[:20],
    }

    print()
    print("Dataset")
    print("-------")
    for cohort, summary in metrics["dataset"]["by_cohort"].items():
        print(
            f"{cohort:<10} rows={summary['rows']} markets={summary['markets']} "
            f"pos={summary['positives']} rate={summary['positive_rate']:.2%}"
        )
    print()
    print("Evaluation (OOF grouped-CV on may_bench / holdout on july_live)")
    print("----------")
    if "prior" in baselines:
        print_metrics("Prior", baselines["prior"]["out_of_fold"])
        print_metrics("Logit", baselines["logistic"]["out_of_fold"])
    print_metrics("Tree OOF", oof_metrics)
    print_metrics("Tree July", holdout_metrics)
    if "logistic" in baselines:
        lift = holdout_metrics["roc_auc"] or 0.0
        base = baselines["logistic"]["holdout"]["roc_auc"] or 0.0
        print()
        print(f"Tree July ROC lift over logistic baseline: {lift - base:+.4f}")

    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
            "config": asdict(config),
            "threshold": threshold,
        },
        args.output_dir / "model.joblib",
    )
    write_json(args.output_dir / "metrics.json", metrics)
    write_json(args.output_dir / "feature_columns.json", feature_columns)
    pd.DataFrame(sentinel).to_csv(
        args.output_dir / "leakage_sentinel.csv", index=False
    )
    save_predictions(
        args.output_dir / "predictions_may_oof.csv",
        train_meta,
        oof["y_prob"].to_numpy(),
        threshold,
        args.target,
    )
    save_predictions(
        args.output_dir / "predictions_july_holdout.csv",
        holdout_meta,
        holdout_prob,
        threshold,
        args.target,
    )

    if not args.skip_permutation_importance and len(np.unique(y_holdout)) == 2:
        importance = permutation_importance(
            model,
            x_holdout,
            y_holdout,
            n_repeats=10,
            random_state=args.seed,
            scoring="average_precision",
        )
        importance_df = pd.DataFrame(
            {
                "feature": feature_columns,
                "importance_mean": importance.importances_mean,
                "importance_std": importance.importances_std,
            }
        ).sort_values("importance_mean", ascending=False)
        importance_df.to_csv(args.output_dir / "permutation_importance_july.csv", index=False)

    print()
    print(f"Saved artifacts to: {args.output_dir}")


if __name__ == "__main__":
    main()
