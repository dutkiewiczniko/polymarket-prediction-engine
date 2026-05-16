from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from scripts.training.train_reward_model import FEATURE_COLUMNS


@dataclass
class RewardModelBundle:
    model: Any
    scaler: Any | None
    feature_columns: list[str]
    source_path: Path

    def predict(self, feature_frame):
        x = feature_frame[self.feature_columns].to_numpy(dtype=np.float32)
        if self.scaler is not None:
            x = self.scaler.transform(x)

        try:
            predictions = self.model.predict(x, verbose=0)
        except TypeError:
            predictions = self.model.predict(x)
        return np.asarray(predictions, dtype=float).reshape(-1)


def _load_tensorflow_model(model_path: Path):
    try:
        import tensorflow as tf
    except ImportError as e:
        raise ImportError("TensorFlow is required to load .keras reward models.") from e

    return tf.keras.models.load_model(model_path)


def _load_json_list(path: Path) -> list[str] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        value = json.load(f)
    if not isinstance(value, list):
        raise ValueError(f"Expected a list in {path}")
    return [str(item) for item in value]


def _bundle_from_directory(path: Path) -> RewardModelBundle:
    model_path = path / "model.keras"
    scaler_path = path / "scaler.joblib"
    feature_columns_path = path / "feature_columns.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model directory does not contain model.keras: {path}")

    scaler = joblib.load(scaler_path) if scaler_path.exists() else None
    feature_columns = _load_json_list(feature_columns_path) or FEATURE_COLUMNS
    return RewardModelBundle(
        model=_load_tensorflow_model(model_path),
        scaler=scaler,
        feature_columns=feature_columns,
        source_path=path,
    )


def _bundle_from_pickle(path: Path) -> RewardModelBundle:
    artifact = joblib.load(path)

    if isinstance(artifact, dict):
        model = artifact.get("model")
        if model is None:
            raise ValueError(f"Pickle bundle must contain a 'model' key: {path}")
        scaler = artifact.get("scaler") or artifact.get("preprocessor")
        feature_columns = artifact.get("feature_columns") or artifact.get("FEATURE_COLUMNS") or FEATURE_COLUMNS
        return RewardModelBundle(
            model=model,
            scaler=scaler,
            feature_columns=[str(column) for column in feature_columns],
            source_path=path,
        )

    if not hasattr(artifact, "predict"):
        raise ValueError(f"Unsupported model artifact; object has no predict() method: {path}")

    return RewardModelBundle(
        model=artifact,
        scaler=None,
        feature_columns=FEATURE_COLUMNS,
        source_path=path,
    )


def _bundle_from_keras_file(path: Path) -> RewardModelBundle:
    scaler_path = path.with_name("scaler.joblib")
    feature_columns_path = path.with_name("feature_columns.json")

    return RewardModelBundle(
        model=_load_tensorflow_model(path),
        scaler=joblib.load(scaler_path) if scaler_path.exists() else None,
        feature_columns=_load_json_list(feature_columns_path) or FEATURE_COLUMNS,
        source_path=path,
    )


def load_reward_model(path: str | Path) -> RewardModelBundle:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Model artifact not found: {path}. Expected a training output directory, .keras file, or .pkl/.joblib bundle."
        )

    if path.is_dir():
        return _bundle_from_directory(path)

    suffix = path.suffix.lower()
    if suffix == ".keras":
        return _bundle_from_keras_file(path)
    if suffix in {".pkl", ".joblib"}:
        return _bundle_from_pickle(path)

    raise ValueError(f"Unsupported model artifact format: {path}")


def find_latest_reward_model(models_dir: str | Path = "models") -> Path | None:
    models_dir = Path(models_dir)
    if not models_dir.exists():
        return None

    candidates = []
    for path in models_dir.iterdir():
        if path.is_dir() and (path / "model.keras").exists():
            candidates.append(path)
        elif path.is_file() and path.suffix.lower() in {".keras", ".pkl", ".joblib"}:
            candidates.append(path)

    if not candidates:
        return None

    return max(candidates, key=lambda path: path.stat().st_mtime)
