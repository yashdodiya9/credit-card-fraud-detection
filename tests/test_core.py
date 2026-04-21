import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from src import data as D
from src.evaluate import best_threshold, threshold_metrics
from src.train import build_models, main
from tests.helpers import make_synthetic


@pytest.fixture(scope="module")
def df():
    return make_synthetic()


def test_time_split_is_ordered_and_disjoint(df):
    sp = D.split_data(df, mode="time")
    assert sp.train["Time"].max() <= sp.val["Time"].min()
    assert sp.val["Time"].max() <= sp.test["Time"].min()
    assert len(sp.train) + len(sp.val) + len(sp.test) == len(df)
    assert not set(sp.train.index) & set(sp.test.index)


def test_stratified_split_keeps_fraud_rate(df):
    sp = D.split_data(df, mode="stratified")
    overall = df[D.TARGET].mean()
    for part in (sp.train, sp.val, sp.test):
        assert abs(part[D.TARGET].mean() - overall) < 0.01


def test_unknown_split_mode_raises(df):
    with pytest.raises(ValueError):
        D.split_data(df, mode="nope")


def test_features_exclude_target_and_raw_time(df):
    X = D.make_features(df)
    assert list(X.columns) == D.FEATURE_COLS
    assert D.TARGET not in X.columns and "Time" not in X.columns
    assert np.isfinite(X.to_numpy()).all()


def test_no_model_cost_equals_total_fraud_value(df):
    y, a = df[D.TARGET].to_numpy(), df["Amount"].to_numpy()
    m = threshold_metrics(y, np.zeros(len(y)), a, threshold=1.0, review_cost=5.0)
    assert m["alerts"] == 0
    assert m["cost"] == pytest.approx(a[y == 1].sum())
    assert m["saving"] == pytest.approx(0.0)


def test_best_threshold_matches_brute_force():
    rng = np.random.default_rng(1)
    y = (rng.random(500) < 0.1).astype(int)
    s = np.clip(rng.normal(0.3 + 0.4 * y, 0.2), 0, 1)
    s = np.round(s, 2)  # create ties on purpose
    a = rng.lognormal(3, 1, 500)
    thr = best_threshold(y, s, a, review_cost=5.0)
    got = threshold_metrics(y, s, a, thr, 5.0)["cost"]
    candidates = np.append(np.unique(s), s.max() + 1)
    brute = min(threshold_metrics(y, s, a, t, 5.0)["cost"] for t in candidates)
    assert got == pytest.approx(brute)


def test_model_beats_random_on_held_out_data(df):
    sp = D.split_data(df, mode="time")
    Xtr, ytr = D.make_features(sp.train), sp.train[D.TARGET].to_numpy()
    Xte, yte = D.make_features(sp.test), sp.test[D.TARGET].to_numpy()
    model = build_models(ytr)["logreg_balanced"].fit(Xtr, ytr)
    ap = average_precision_score(yte, model.predict_proba(Xte)[:, 1])
    assert ap > 3 * yte.mean()


def test_smote_runs_inside_pipeline_only():
    pytest.importorskip("imblearn")
    from imblearn.pipeline import Pipeline as ImbPipeline

    models = build_models(np.array([0] * 95 + [1] * 5))
    assert isinstance(models["logreg_smote"], ImbPipeline)


def test_end_to_end_training_writes_artifacts(tmp_path, df):
    csv = tmp_path / "creditcard.csv"
    df.to_csv(csv, index=False)
    rc = main([
        "--data", str(csv), "--models", "logreg_balanced,random_forest_balanced",
        "--out-dir", str(tmp_path / "models"), "--reports-dir", str(tmp_path / "reports"),
        "--n-boot", "30", "--readme", str(tmp_path / "none.md"),
    ])
    assert rc == 0
    assert (tmp_path / "models" / "model.joblib").exists()
    assert (tmp_path / "reports" / "metrics.json").exists()
    assert (tmp_path / "reports" / "figures" / "pr_curves.png").exists()


def test_api_roundtrip(tmp_path, df):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from src.api import create_app

    csv = tmp_path / "creditcard.csv"
    df.to_csv(csv, index=False)
    main(["--data", str(csv), "--models", "logreg_balanced", "--out-dir", str(tmp_path / "m"),
          "--reports-dir", str(tmp_path / "r"), "--n-boot", "10", "--readme", str(tmp_path / "x.md")])
    row = df.drop(columns=[D.TARGET]).iloc[0].to_dict()
    with TestClient(create_app(str(tmp_path / "m" / "model.joblib"))) as client:
        assert client.get("/health").json()["status"] == "ok"
        body = client.post("/predict", json=row).json()
        assert body["decision"] in {"approve", "review", "block"}
        assert 0.0 <= body["fraud_probability"] <= 1.0
        bad = dict(row)
        bad["Amount"] = -5
        assert client.post("/predict", json=bad).status_code == 422
