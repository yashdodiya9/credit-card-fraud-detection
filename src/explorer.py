"""Pure functions behind the Streamlit explorer (kept free of UI code so they can be tested)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

SCORE_PREFIX = "score_"


def load_scores(path: str | Path) -> pd.DataFrame:
    """Read reports/scores.csv.gz written by ``python -m src.train``."""
    df = pd.read_csv(path)
    needed = {"split", "tx_id", "Time", "Amount", "Class"}
    if not needed.issubset(df.columns):
        raise ValueError(f"Scores file is missing columns: {sorted(needed - set(df.columns))}")
    return df


def model_names(df: pd.DataFrame) -> list[str]:
    return [c[len(SCORE_PREFIX):] for c in df.columns if c.startswith(SCORE_PREFIX)]


def threshold_for_alert_rate(scores, alerts_per_10k: float) -> float:
    """Score threshold that flags about ``alerts_per_10k`` of every 10,000 transactions."""
    s = np.sort(np.asarray(scores, dtype=float))[::-1]
    k = int(round(alerts_per_10k / 10_000 * len(s)))
    if k <= 0:
        return float(np.nextafter(s[0], np.inf))
    return float(s[min(k, len(s)) - 1])


def cost_curve(y_true, scores, amounts, review_cost: float, n_points: int = 300) -> pd.DataFrame:
    """Operating curve: for many thresholds, workload (alerts) vs protection (fraud value caught)."""
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    a = np.asarray(amounts, dtype=float)
    n = len(s)
    order = np.argsort(-s, kind="stable")
    s_desc = s[order]
    fraud_amt = a[order] * (y[order] == 1)
    tp_cum = np.concatenate([[0], np.cumsum(y[order] == 1)])
    caught_cum = np.concatenate([[0.0], np.cumsum(fraud_amt)])
    total_fraud, total_n_fraud = caught_cum[-1], tp_cum[-1]

    q = 1 - np.logspace(-4.5, 0, n_points)  # dense in the top tail where alerts live
    thresholds = np.unique(np.quantile(s, q))
    asc = s_desc[::-1]
    k = n - np.searchsorted(asc, thresholds, side="left")  # number flagged at each threshold
    cost = total_fraud - caught_cum[k] + review_cost * k
    return pd.DataFrame(
        {
            "threshold": thresholds,
            "alerts_per_10k": k / n * 10_000,
            "precision": np.where(k > 0, tp_cum[k] / np.maximum(k, 1), 0.0),
            "recall": tp_cum[k] / total_n_fraud if total_n_fraud else 0.0,
            "value_caught_pct": 100 * caught_cum[k] / total_fraud if total_fraud else 0.0,
            "cost": cost,
            "saving_pct": 100 * (total_fraud - cost) / total_fraud if total_fraud else 0.0,
        }
    )


def band_summary(y_true, scores, amounts, review_threshold: float, block_threshold: float):
    """Approve / review / block bands with counts and fraud inside each band."""
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    a = np.asarray(amounts, dtype=float)
    block_threshold = max(block_threshold, review_threshold)
    band = np.where(s >= block_threshold, "block", np.where(s >= review_threshold, "review", "approve"))
    rows = []
    for name in ("approve", "review", "block"):
        m = band == name
        rows.append(
            {
                "band": name,
                "transactions": int(m.sum()),
                "share_%": 100 * float(m.mean()),
                "frauds": int((y[m] == 1).sum()),
                "fraud_value": float(a[m & (y == 1)].sum()),
                "precision_%": 100 * float((y[m] == 1).mean()) if m.any() else 0.0,
            }
        )
    return pd.DataFrame(rows), band
