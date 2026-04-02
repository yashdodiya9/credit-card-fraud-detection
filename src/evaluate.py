"""Evaluation helpers: ranking metrics, cost-based thresholding, plots."""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    average_precision_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)


def best_threshold(y_true, scores, amounts, review_cost: float) -> float:
    """Score threshold that minimises total expected cost.

    cost = value of fraud we fail to flag + review_cost * number of alerts.
    Only thresholds at tie-group boundaries are considered, so the returned
    threshold reproduces exactly the cost it was selected for.
    """
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    a = np.asarray(amounts, dtype=float)
    order = np.argsort(-s, kind="stable")
    y_s, a_s, s_s = y[order], a[order], s[order]
    fraud_amt = a_s * (y_s == 1)
    total = fraud_amt.sum()
    caught = np.concatenate([[0.0], np.cumsum(fraud_amt)])
    n = len(s)
    k = np.arange(n + 1)
    cost = total - caught + review_cost * k
    valid = np.ones(n + 1, dtype=bool)
    if n > 1:
        valid[1:n] = s_s[:-1] > s_s[1:]
    cost = np.where(valid, cost, np.inf)
    k_best = int(np.argmin(cost))
    if k_best == 0:  # flagging nothing is cheapest
        return float(np.nextafter(s_s[0], np.inf))
    return float(s_s[k_best - 1])


def block_threshold(scores, review_threshold: float, share: float = 0.3) -> float:
    """Threshold above which alerts are auto-blocked instead of sent to a human.

    Among scores at or above the review threshold, the highest-scoring ``share``
    are blocked; the rest go to manual review. This is a policy choice, not a
    statistically optimal value.
    """
    s = np.asarray(scores, dtype=float)
    flagged = s[s >= review_threshold]
    if flagged.size == 0 or share >= 1:
        return float(review_threshold)
    if share <= 0:
        return float(np.nextafter(flagged.max(), np.inf))
    return float(max(np.quantile(flagged, 1 - share), review_threshold))


def threshold_metrics(y_true, scores, amounts, threshold: float, review_cost: float) -> dict:
    """Operational metrics at a fixed threshold."""
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    a = np.asarray(amounts, dtype=float)
    flagged = s >= threshold
    fraud = y == 1
    total_fraud_amt = float(a[fraud].sum())
    caught_amt = float(a[fraud & flagged].sum())
    missed_amt = total_fraud_amt - caught_amt
    review = float(flagged.sum()) * review_cost
    cost = missed_amt + review
    baseline = total_fraud_amt  # no model: every fraud goes through
    return {
        "precision": float(precision_score(y, flagged, zero_division=0)),
        "recall": float(recall_score(y, flagged, zero_division=0)),
        "alerts": int(flagged.sum()),
        "alerts_per_10k": float(flagged.mean() * 10_000),
        "fraud_value_total": total_fraud_amt,
        "fraud_value_caught": caught_amt,
        "value_caught_pct": 100.0 * caught_amt / total_fraud_amt if total_fraud_amt else 0.0,
        "cost": cost,
        "baseline_cost": baseline,
        "saving": baseline - cost,
        "saving_pct": 100.0 * (baseline - cost) / baseline if baseline else 0.0,
    }


def bootstrap_ci(y_true, scores, n_boot: int = 500, seed: int = 0, alpha: float = 0.05):
    """Bootstrap confidence interval for PR-AUC (average precision)."""
    y = np.asarray(y_true)
    s = np.asarray(scores, dtype=float)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        if y[idx].sum() == 0:
            continue
        vals.append(average_precision_score(y[idx], s[idx]))
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def plot_pr_curves(curves: dict, path: Path) -> None:
    """curves: name -> (y_true, scores)."""
    fig, ax = plt.subplots(figsize=(7, 5))
    prevalence = None
    for name, (y, s) in curves.items():
        p, r, _ = precision_recall_curve(y, s)
        ax.plot(r, p, label=f"{name} (AP={average_precision_score(y, s):.3f})")
        prevalence = float(np.mean(y))
    if prevalence is not None:
        ax.axhline(prevalence, ls="--", c="grey", label=f"random ({prevalence:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curves (test set)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confusion(y_true, scores, threshold: float, path: Path, title: str) -> None:
    y = np.asarray(y_true)
    flagged = np.asarray(scores) >= threshold
    cm = np.array(
        [
            [np.sum((y == 0) & ~flagged), np.sum((y == 0) & flagged)],
            [np.sum((y == 1) & ~flagged), np.sum((y == 1) & flagged)],
        ]
    )
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center")
    ax.set_xticks([0, 1], ["Not flagged", "Flagged"])
    ax.set_yticks([0, 1], ["Legit", "Fraud"])
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
