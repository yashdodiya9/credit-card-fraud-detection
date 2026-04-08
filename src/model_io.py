"""Save / load the trained model together with its decision threshold."""
from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd

from .data import make_features


def save_artifact(path: Path, model, threshold: float, meta: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": float(threshold), "meta": meta}, path)


def load_artifact(path: Path) -> dict:
    return joblib.load(path)


def score_transactions(artifact: dict, raw: pd.DataFrame):
    """Fraud probability for raw transaction rows (Time, V1..V28, Amount)."""
    return artifact["model"].predict_proba(make_features(raw))[:, 1]
