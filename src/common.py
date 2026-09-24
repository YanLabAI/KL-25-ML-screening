"""Shared utilities for the KL-25 machine-learning workflow."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.metrics import pairwise_distances

RANDOM_STATE = 42
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def safe_dump(obj: Any, path: str | Path) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    if path.exists():
        path.unlink()
    joblib.dump(obj, path)


def read_csv_checked(path: str | Path, required_columns: Iterable[str]) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input file not found: {path}")
    df = pd.read_csv(path)
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required column(s) {missing} in {path}")
    return df


def clean_sequence(value: Any) -> str:
    if pd.isna(value):
        return ""
    return "".join(str(value).split()).upper()


def sequence_is_standard(sequence: str) -> bool:
    return bool(sequence) and set(sequence).issubset(STANDARD_AA)


def numeric_feature_frame(
    df: pd.DataFrame,
    exclude: Iterable[str],
) -> pd.DataFrame:
    excluded = set(exclude)
    feature_df = df[[c for c in df.columns if c not in excluded]].copy()
    non_numeric = [c for c in feature_df.columns if not pd.api.types.is_numeric_dtype(feature_df[c])]
    if non_numeric:
        raise ValueError(
            "All model features must be numeric. Non-numeric columns found: "
            + ", ".join(non_numeric[:20])
        )
    return feature_df


def kennard_stone(X: pd.DataFrame | np.ndarray, ratio: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """Kennard-Stone split matching the logic used in the original analysis."""
    if not 0 < ratio < 1:
        raise ValueError("ratio must be between 0 and 1")
    values = np.asarray(X, dtype=float)
    n_samples = values.shape[0]
    if n_samples < 3:
        raise ValueError("Kennard-Stone split requires at least 3 samples")
    n_train = max(2, min(n_samples - 1, int(n_samples * ratio)))

    distances = pairwise_distances(values, metric="euclidean")
    first_pair = np.unravel_index(np.argmax(distances), distances.shape)
    selected = list(dict.fromkeys(first_pair))
    remaining = [i for i in range(n_samples) if i not in selected]

    while len(selected) < n_train and remaining:
        dmin = np.min(distances[np.ix_(remaining, selected)], axis=1)
        next_idx = remaining[int(np.argmax(dmin))]
        selected.append(next_idx)
        remaining.remove(next_idx)

    return np.asarray(selected, dtype=int), np.asarray(remaining, dtype=int)


def classification_metrics(y_true, y_pred, y_prob) -> dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_weighted": float(precision_score(y_true, y_pred, average="weighted", zero_division=0)),
        "recall_weighted": float(recall_score(y_true, y_pred, average="weighted", zero_division=0)),
        "f1_weighted": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "auc": float(roc_auc_score(y_true, y_prob)),
    }


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
    }


def usable_cores(fraction: float = 0.85) -> int:
    total = os.cpu_count() or 1
    return max(1, int(total * fraction))
