"""Synthetic data with the same schema as the Kaggle creditcard.csv (used by tests and CI)."""
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from src.data import TARGET, V_COLS


def make_synthetic(n: int = 6000, fraud_rate: float = 0.02, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X, y = make_classification(
        n_samples=n, n_features=28, n_informative=12, n_redundant=4,
        weights=[1 - fraud_rate, fraud_rate], class_sep=1.5, flip_y=0.0, random_state=seed,
    )
    df = pd.DataFrame(X, columns=V_COLS)
    df.insert(0, "Time", np.sort(rng.uniform(0, 172_800, n)))
    df["Amount"] = np.round(rng.lognormal(3.5, 1.2, n), 2)
    df[TARGET] = y
    return df
