"""Data loading, splitting and feature engineering.

Everything here is shared by training and serving, so the API and the
trained model can never disagree about how features are built.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

V_COLS = [f"V{i}" for i in range(1, 29)]
RAW_COLS = ["Time", *V_COLS, "Amount"]
TARGET = "Class"
FEATURE_COLS = [*V_COLS, "log_amount", "hour_sin", "hour_cos"]
MIN_POSITIVES_PER_SPLIT = 5


@dataclass
class Splits:
    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame


def load_data(path: str) -> pd.DataFrame:
    """Read the Kaggle creditcard.csv and validate its schema."""
    df = pd.read_csv(path)
    needed = [*RAW_COLS, TARGET]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset is missing columns: {missing}")
    if df[needed].isna().any().any():
        raise ValueError("Dataset contains missing values; this pipeline expects none.")
    return df


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """Build model features from raw transaction columns.

    - Raw ``Time`` (seconds since the first transaction) is NOT used directly:
      it cannot generalise to future transactions. We keep only hour-of-day,
      encoded cyclically.
    - ``Amount`` is heavy-tailed, so it is log-transformed.
    """
    out = df[V_COLS].copy()
    out["log_amount"] = np.log1p(df["Amount"].clip(lower=0))
    hour = (df["Time"] // 3600) % 24
    angle = 2 * np.pi * hour / 24
    out["hour_sin"] = np.sin(angle)
    out["hour_cos"] = np.cos(angle)
    return out[FEATURE_COLS]


def split_data(
    df: pd.DataFrame,
    mode: str = "time",
    val_frac: float = 0.2,
    test_frac: float = 0.2,
    seed: int = 42,
) -> Splits:
    """Split into train / validation / test.

    ``time``: train on the earliest transactions, validate on the next block,
    test on the latest block (how a deployed model is actually used).
    ``stratified``: random split that preserves the fraud rate.
    """
    if mode == "time":
        ordered = df.sort_values("Time", kind="stable")
        n = len(ordered)
        n_train = int(n * (1 - val_frac - test_frac))
        n_val = int(n * (1 - test_frac))
        train, val, test = ordered.iloc[:n_train], ordered.iloc[n_train:n_val], ordered.iloc[n_val:]
    elif mode == "stratified":
        rest, test = train_test_split(
            df, test_size=test_frac, stratify=df[TARGET], random_state=seed
        )
        train, val = train_test_split(
            rest, test_size=val_frac / (1 - test_frac), stratify=rest[TARGET], random_state=seed
        )
    else:
        raise ValueError(f"Unknown split mode: {mode!r} (use 'time' or 'stratified')")

    for name, part in (("train", train), ("val", val), ("test", test)):
        if int(part[TARGET].sum()) < MIN_POSITIVES_PER_SPLIT:
            raise ValueError(f"{name} split has fewer than {MIN_POSITIVES_PER_SPLIT} fraud cases.")
    return Splits(train=train, val=val, test=test)
