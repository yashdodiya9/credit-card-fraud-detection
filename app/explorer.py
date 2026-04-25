"""Streamlit explorer: trade fraud caught against analyst workload.

Run from the repo root:  streamlit run app/explorer.py
Needs reports/scores.csv.gz  (created by:  python -m src.train --data data/creditcard.csv)
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from src.evaluate import best_threshold, block_threshold, threshold_metrics  # noqa: E402
from src.explorer import (  # noqa: E402
    band_summary,
    cost_curve,
    load_scores,
    model_names,
    threshold_for_alert_rate,
)

SCORES = ROOT / "reports" / "scores.csv.gz"
METRICS = ROOT / "reports" / "metrics.json"

st.set_page_config(page_title="Fraud Threshold Explorer", layout="wide")
st.title("Fraud alert threshold explorer")
st.caption(
    "Every number is computed on a held-out **test** set. Thresholds are tuned on the **validation** "
    "set, as in training. The review cost is an assumption you control, not a measured figure."
)


@st.cache_data
def get_scores(path: str) -> pd.DataFrame:
    return load_scores(path)


if not SCORES.exists():
    st.error("reports/scores.csv.gz not found. Run `python -m src.train --data data/creditcard.csv` first.")
    st.stop()

df = get_scores(str(SCORES))
val, test = df[df["split"] == "val"], df[df["split"] == "test"]
models = model_names(df)
default_model = 0
if METRICS.exists():
    selected = json.loads(METRICS.read_text())["meta"].get("model_name")
    if selected in models:
        default_model = models.index(selected)

with st.sidebar:
    st.header("Controls")
    model = st.selectbox("Model", models, index=default_model)
    review_cost = st.slider("Review cost per alert", min_value=0.5, max_value=100.0, value=5.0, step=0.5)
    mode = st.radio("Threshold", ["Cost-optimal (tuned on validation)", "Set alert volume manually"])
    block_share = st.slider("Auto-block share of alerts", min_value=0, max_value=100, value=30, step=5)

col = f"score_{model}"
yv, sv, av = val["Class"], val[col], val["Amount"]
yt, stt, at = test["Class"], test[col], test["Amount"]

optimal = best_threshold(yv, sv, av, review_cost)
if mode.startswith("Cost-optimal"):
    thr = optimal
else:
    optimal_rate = float((stt >= optimal).mean() * 10_000)
    rate = st.sidebar.slider(
        "Alerts per 10,000 transactions", min_value=0.0, max_value=100.0,
        value=float(round(optimal_rate, 1)), step=0.5,
    )
    thr = threshold_for_alert_rate(sv, rate)

m = threshold_metrics(yt, stt, at, thr, review_cost)
block_thr = block_threshold(sv, thr, block_share / 100)

tab1, tab2, tab3 = st.tabs(["Trade-off", "Decision bands & review queue", "Model comparison"])

with tab1:
    c = st.columns(5)
    c[0].metric("Precision", f"{m['precision']:.2f}")
    c[1].metric("Recall", f"{m['recall']:.2f}")
    c[2].metric("Alerts / 10k txns", f"{m['alerts_per_10k']:.1f}")
    c[3].metric("Fraud value caught", f"{m['value_caught_pct']:.1f}%")
    c[4].metric("Cost saving vs no model", f"{m['saving_pct']:.1f}%")
    st.write(f"Threshold: `{thr:.6f}`  |  frauds in test set: **{int((yt == 1).sum())}**")

    curve = cost_curve(yt, stt, at, review_cost)
    left, right = st.columns(2)
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.plot(curve["alerts_per_10k"], curve["saving_pct"])
    ax.scatter([m["alerts_per_10k"]], [m["saving_pct"]], c="red", zorder=3, label="current setting")
    ax.set_xlabel("Alerts per 10,000 transactions")
    ax.set_ylabel("Cost saving vs no model (%)")
    ax.set_title("Net benefit vs analyst workload")
    ax.legend()
    left.pyplot(fig)
    fig2, ax2 = plt.subplots(figsize=(5.5, 3.8))
    ax2.plot(curve["alerts_per_10k"], curve["value_caught_pct"])
    ax2.scatter([m["alerts_per_10k"]], [m["value_caught_pct"]], c="red", zorder=3)
    ax2.set_xlabel("Alerts per 10,000 transactions")
    ax2.set_ylabel("Fraud value caught (%)")
    ax2.set_title("Protection vs analyst workload")
    right.pyplot(fig2)
    st.info(
        "With only a few dozen fraud cases in the test set, small moves along these curves are noise. "
        "Look at the overall shape, not single points."
    )

with tab2:
    summary, band = band_summary(yt, stt, at, thr, block_thr)
    st.dataframe(summary.round(2), hide_index=True)
    st.caption(
        "approve = below review threshold; review = sent to a human analyst; "
        "block = highest-scoring alerts stopped automatically."
    )
    queue = test.loc[band == "review", ["tx_id", "Time", "Amount", "Class", col]].copy()
    queue["hour_of_day"] = ((queue["Time"] // 3600) % 24).astype(int)
    queue = queue.sort_values(col, ascending=False).head(50)
    queue = queue.rename(columns={col: "score", "Class": "actual_fraud"})
    queue["decision"] = "pending"
    st.subheader(f"Review queue (top {len(queue)} by score)")
    edited = st.data_editor(
        queue[["tx_id", "hour_of_day", "Amount", "score", "decision", "actual_fraud"]],
        hide_index=True,
        column_config={
            "decision": st.column_config.SelectboxColumn(options=["pending", "approve", "block"]),
        },
        disabled=["tx_id", "hour_of_day", "Amount", "score", "actual_fraud"],
    )
    st.caption(
        "Demo only: decisions are not saved, and `actual_fraud` is shown for illustration. "
        "A real analyst would not see the true label."
    )
    decided = edited[edited["decision"] != "pending"]
    if len(decided):
        correct = ((decided["decision"] == "block") == (decided["actual_fraud"] == 1)).mean()
        st.write(f"You decided {len(decided)} cases; {correct:.0%} matched the true label.")

with tab3:
    rows = []
    for name in models:
        vv, tt = val[f"score_{name}"], test[f"score_{name}"]
        t_opt = best_threshold(yv, vv, av, review_cost)
        mm = threshold_metrics(yt, tt, at, t_opt, review_cost)
        rows.append({
            "model": name, "precision": mm["precision"], "recall": mm["recall"],
            "alerts_per_10k": mm["alerts_per_10k"], "value_caught_%": mm["value_caught_pct"],
            "cost_saving_%": mm["saving_pct"],
        })
    st.dataframe(pd.DataFrame(rows).round(3), hide_index=True)
    st.caption(
        "Each model uses its own validation-tuned threshold at the review cost chosen in the sidebar. "
        "Differences between models are often smaller than the test-set uncertainty."
    )
