"""Train, compare and select fraud models; write metrics, plots and a model artifact.

Usage:
    python -m src.train --data data/creditcard.csv --update-readme
"""
from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from . import data as D
from .evaluate import (
    best_threshold,
    block_threshold,
    bootstrap_ci,
    plot_confusion,
    plot_pr_curves,
    threshold_metrics,
)
from .model_io import save_artifact

try:
    from xgboost import XGBClassifier
except ImportError:  # optional at import time; listed in requirements.txt
    XGBClassifier = None

try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
except ImportError:
    SMOTE = None
    ImbPipeline = None

README_START = "<!-- RESULTS:START -->"
README_END = "<!-- RESULTS:END -->"


def _xgb(seed: int, scale_pos_weight: float):
    return XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        tree_method="hist",
        n_jobs=-1,
        random_state=seed,
    )


def build_models(y_train, seed: int = 42) -> dict:
    """Candidate models. SMOTE variants resample inside an imblearn Pipeline,
    so resampling only ever touches training data, never validation/test."""
    y_train = np.asarray(y_train)
    neg, pos = int((y_train == 0).sum()), int((y_train == 1).sum())
    spw = neg / max(pos, 1)

    models = {
        "logreg_balanced": Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "clf",
                    LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
                ),
            ]
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced_subsample",
            n_jobs=-1,
            random_state=seed,
        ),
    }
    if XGBClassifier is not None:
        models["xgboost_weighted"] = _xgb(seed, spw)
    if ImbPipeline is not None:
        models["logreg_smote"] = ImbPipeline(
            [
                ("scale", StandardScaler()),
                ("smote", SMOTE(sampling_strategy=0.1, random_state=seed)),
                ("clf", LogisticRegression(max_iter=2000, random_state=seed)),
            ]
        )
        if XGBClassifier is not None:
            models["xgboost_smote"] = ImbPipeline(
                [
                    ("smote", SMOTE(sampling_strategy=0.1, random_state=seed)),
                    ("clf", _xgb(seed, 1.0)),
                ]
            )
    return models


def _fmt_table(rows: list[dict]) -> str:
    cols = [
        ("model", "Model", "{}"),
        ("val_pr_auc", "Val PR-AUC", "{:.3f}"),
        ("test_pr_auc", "Test PR-AUC", "{:.3f}"),
        ("ci", "95% CI", "{}"),
        ("test_roc_auc", "Test ROC-AUC", "{:.3f}"),
        ("precision", "Precision", "{:.2f}"),
        ("recall", "Recall", "{:.2f}"),
        ("alerts_per_10k", "Alerts/10k txns", "{:.1f}"),
        ("value_caught_pct", "Fraud value caught %", "{:.1f}"),
        ("saving_pct", "Cost saving vs no model %", "{:.1f}"),
    ]
    head = "| " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(fmt.format(r[key]) for key, _, fmt in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def export_scores(path: Path, frames: dict, scores: dict) -> None:
    """Save validation and test scores for every model (used by the Streamlit explorer)."""
    parts = []
    for split_name, frame in frames.items():
        part = frame[["Time", "Amount", D.TARGET]].copy()
        part.insert(0, "split", split_name)
        part.insert(1, "tx_id", frame.index)
        for model_name, per_split in scores.items():
            part[f"score_{model_name}"] = per_split[split_name]
        parts.append(part)
    out = pd.concat(parts, ignore_index=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False, float_format="%.7g")


def update_readme(readme: Path, block: str) -> bool:
    if not readme.exists():
        return False
    text = readme.read_text(encoding="utf-8")
    pattern = re.compile(re.escape(README_START) + r".*?" + re.escape(README_END), re.S)
    if not pattern.search(text):
        return False
    new = f"{README_START}\n{block}\n{README_END}"
    readme.write_text(pattern.sub(lambda _m: new, text), encoding="utf-8")
    return True


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="data/creditcard.csv")
    ap.add_argument("--split", choices=["time", "stratified"], default="time")
    ap.add_argument("--review-cost", type=float, default=5.0,
                    help="Assumed cost of manually reviewing one alert (same units as Amount).")
    ap.add_argument("--models", default="", help="Comma-separated subset of model names.")
    ap.add_argument("--select-by", choices=["cost", "pr_auc"], default="cost",
                    help="Validation criterion for choosing the final model.")
    ap.add_argument("--block-share", type=float, default=0.3,
                    help="Share of highest-scoring alerts that are auto-blocked (rest go to review).")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out-dir", default="models")
    ap.add_argument("--reports-dir", default="reports")
    ap.add_argument("--readme", default="README.md")
    ap.add_argument("--update-readme", action="store_true")
    args = ap.parse_args(argv)

    out_dir, rep_dir = Path(args.out_dir), Path(args.reports_dir)
    fig_dir = rep_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = D.load_data(args.data)
    sp = D.split_data(df, mode=args.split, seed=args.seed)
    Xtr, ytr = D.make_features(sp.train), sp.train[D.TARGET].to_numpy()
    Xva, yva = D.make_features(sp.val), sp.val[D.TARGET].to_numpy()
    Xte, yte = D.make_features(sp.test), sp.test[D.TARGET].to_numpy()
    ava, ate = sp.val["Amount"].to_numpy(), sp.test["Amount"].to_numpy()

    models = build_models(ytr, seed=args.seed)
    wanted = [m.strip() for m in args.models.split(",") if m.strip()]
    if wanted:
        unknown = [m for m in wanted if m not in models]
        if unknown:
            raise SystemExit(f"Unknown/unavailable models: {unknown}. Available: {list(models)}")
        models = {k: v for k, v in models.items() if k in wanted}

    print(f"Split={args.split}  train={len(sp.train):,} (fraud {ytr.sum()})  "
          f"val={len(sp.val):,} (fraud {yva.sum()})  test={len(sp.test):,} (fraud {yte.sum()})")

    rows, fitted, test_scores, all_scores = [], {}, {}, {}
    for name, model in models.items():
        t0 = time.time()
        model.fit(Xtr, ytr)
        fit_s = time.time() - t0
        s_val = model.predict_proba(Xva)[:, 1]
        s_test = model.predict_proba(Xte)[:, 1]
        thr = best_threshold(yva, s_val, ava, args.review_cost)  # tuned on validation only
        m = threshold_metrics(yte, s_test, ate, thr, args.review_cost)
        val_m = threshold_metrics(yva, s_val, ava, thr, args.review_cost)
        lo, hi = bootstrap_ci(yte, s_test, n_boot=args.n_boot, seed=args.seed)
        row = {
            "model": name,
            "val_pr_auc": float(average_precision_score(yva, s_val)),
            "val_saving_pct": val_m["saving_pct"],
            "test_pr_auc": float(average_precision_score(yte, s_test)),
            "ci": f"{lo:.3f}-{hi:.3f}",
            "test_roc_auc": float(roc_auc_score(yte, s_test)),
            "threshold": thr,
            "fit_seconds": fit_s,
            **m,
        }
        rows.append(row)
        fitted[name] = model
        test_scores[name] = s_test
        all_scores[name] = {"val": s_val, "test": s_test}
        print(f"{name:24s} val PR-AUC={row['val_pr_auc']:.3f}  test PR-AUC={row['test_pr_auc']:.3f}  "
              f"recall={m['recall']:.2f}  precision={m['precision']:.2f}  ({fit_s:.0f}s)")

    key = "val_saving_pct" if args.select_by == "cost" else "val_pr_auc"
    best = max(rows, key=lambda r: r[key])  # selection uses validation, never test
    best_name = best["model"]

    plot_pr_curves({n: (yte, s) for n, s in test_scores.items()}, fig_dir / "pr_curves.png")
    plot_confusion(yte, test_scores[best_name], best["threshold"], fig_dir / "confusion_matrix.png",
                   f"{best_name} @ cost-optimal threshold (test)")

    best_block = block_threshold(all_scores[best_name]["val"], best["threshold"], args.block_share)
    meta = {
        "model_name": best_name,
        "select_by": args.select_by,
        "review_threshold": float(best["threshold"]),
        "block_threshold": best_block,
        "block_share": args.block_share,
        "split": args.split,
        "review_cost": args.review_cost,
        "features": D.FEATURE_COLS,
        "n_train": len(sp.train), "n_val": len(sp.val), "n_test": len(sp.test),
        "frauds_test": int(yte.sum()),
    }
    save_artifact(out_dir / "model.joblib", fitted[best_name], best["threshold"], meta)
    export_scores(rep_dir / "scores.csv.gz", {"val": sp.val, "test": sp.test}, all_scores)
    (rep_dir / "metrics.json").write_text(
        json.dumps({"meta": meta, "models": rows}, indent=2), encoding="utf-8")

    block = "\n".join([
        f"Split: **{args.split}** | train {len(sp.train):,} / val {len(sp.val):,} / test {len(sp.test):,} "
        f"transactions | frauds in test: **{int(yte.sum())}** | assumed review cost per alert: "
        f"**{args.review_cost:g}**",
        "",
        _fmt_table(rows),
        "",
        f"Selected model (best validation {'cost saving' if args.select_by == 'cost' else 'PR-AUC'}): "
        f"**{best_name}**. Thresholds are tuned on the "
        "validation set only; all numbers above are on the held-out test set. The test set has few "
        "fraud cases, so read the confidence interval, not just the point estimate.",
        "",
        "![PR curves](reports/figures/pr_curves.png)",
    ])
    (rep_dir / "results.md").write_text(block + "\n", encoding="utf-8")
    if args.update_readme and update_readme(Path(args.readme), block):
        print(f"Updated {args.readme}")
    print(f"\nSelected: {best_name}  threshold={best['threshold']:.4f}  -> {out_dir / 'model.joblib'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
