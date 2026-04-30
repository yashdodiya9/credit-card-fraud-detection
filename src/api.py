"""FastAPI service that scores a transaction and returns an approve / review / block decision.

Run:  uvicorn src.api:app --reload
"""
from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field, create_model

from .data import V_COLS
from .model_io import load_artifact, score_transactions

_fields = {c: (float, Field(..., description="Anonymised PCA component")) for c in V_COLS}
Transaction = create_model(
    "Transaction",
    Time=(float, Field(..., ge=0, description="Seconds since first transaction in the dataset")),
    Amount=(float, Field(..., ge=0, description="Transaction amount")),
    **_fields,
)


class PredictionResponse(BaseModel):
    fraud_probability: float
    decision: Literal["approve", "review", "block"]
    review_threshold: float
    block_threshold: float
    scoring_model: str


def create_app(model_path: str | None = None) -> FastAPI:
    path = Path(model_path or os.getenv("MODEL_PATH", "models/model.joblib"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if not path.exists():
            raise RuntimeError(f"Model file not found at {path}. Run `python -m src.train` first.")
        app.state.artifact = load_artifact(path)
        yield

    app = FastAPI(title="Credit Card Fraud Scoring API", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health():
        meta = app.state.artifact["meta"]
        return {"status": "ok", "model": meta["model_name"], "split": meta["split"]}

    @app.post("/predict", response_model=PredictionResponse)
    def predict(tx: Transaction):  # type: ignore[valid-type]
        art = app.state.artifact
        prob = float(score_transactions(art, pd.DataFrame([tx.model_dump()]))[0])
        review_thr = float(art["threshold"])
        # Policy knob: the most confident alerts are blocked outright; the rest go to a human.
        # Set BLOCK_THRESHOLD to override the value stored at training time.
        default_block = art["meta"].get("block_threshold", review_thr + 0.5 * (1 - review_thr))
        block_thr = max(float(os.getenv("BLOCK_THRESHOLD", default_block)), review_thr)
        decision = "block" if prob >= block_thr else "review" if prob >= review_thr else "approve"
        return PredictionResponse(
            fraud_probability=prob,
            decision=decision,
            review_threshold=review_thr,
            block_threshold=block_thr,
            scoring_model=art["meta"]["model_name"],
        )

    return app


app = create_app()
