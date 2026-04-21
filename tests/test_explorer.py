import numpy as np
import pytest

from src import data as D
from src.evaluate import block_threshold, threshold_metrics
from src.explorer import (
    band_summary,
    cost_curve,
    load_scores,
    model_names,
    threshold_for_alert_rate,
)
from src.train import main
from tests.helpers import make_synthetic


def _toy(seed=3, n=2000):
    rng = np.random.default_rng(seed)
    y = (rng.random(n) < 0.05).astype(int)
    s = np.clip(rng.normal(0.2 + 0.5 * y, 0.2), 0, 1)
    a = rng.lognormal(3, 1, n)
    return y, s, a


def test_cost_curve_matches_threshold_metrics():
    y, s, a = _toy()
    curve = cost_curve(y, s, a, review_cost=5.0)
    for i in (5, len(curve) // 2, len(curve) - 5):
        row = curve.iloc[i]
        m = threshold_metrics(y, s, a, row["threshold"], 5.0)
        assert row["cost"] == pytest.approx(m["cost"], rel=1e-9)
        assert row["recall"] == pytest.approx(m["recall"])
        assert row["alerts_per_10k"] == pytest.approx(m["alerts_per_10k"])


def test_threshold_for_alert_rate_hits_target_rate():
    _, s, _ = _toy()
    thr = threshold_for_alert_rate(s, 100)  # 1% of transactions
    assert abs((s >= thr).mean() - 0.01) < 0.002


def test_band_summary_partitions_all_transactions():
    y, s, a = _toy()
    summary, band = band_summary(y, s, a, review_threshold=0.5, block_threshold=0.7)
    assert summary["transactions"].sum() == len(y)
    assert summary["frauds"].sum() == y.sum()
    assert set(band) <= {"approve", "review", "block"}


def test_block_threshold_blocks_requested_share_of_alerts():
    _, s, _ = _toy()
    review = 0.5
    thr = block_threshold(s, review, share=0.3)
    flagged = s[s >= review]
    assert abs((flagged >= thr).mean() - 0.3) < 0.02
    assert block_threshold(s, review, share=0.0) > s.max()
    assert block_threshold(s, review, share=1.0) == review


def test_training_exports_scores_for_explorer(tmp_path):
    df = make_synthetic()
    csv = tmp_path / "creditcard.csv"
    df.to_csv(csv, index=False)
    main(["--data", str(csv), "--models", "logreg_balanced,random_forest_balanced",
          "--out-dir", str(tmp_path / "m"), "--reports-dir", str(tmp_path / "r"),
          "--n-boot", "10", "--readme", str(tmp_path / "x.md")])
    scores = load_scores(tmp_path / "r" / "scores.csv.gz")
    assert set(scores["split"]) == {"val", "test"}
    assert model_names(scores) == ["logreg_balanced", "random_forest_balanced"]
    assert scores["Class"].isin([0, 1]).all()
    assert D.TARGET in scores.columns
