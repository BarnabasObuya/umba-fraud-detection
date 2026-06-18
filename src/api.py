"""
api.py — Umba Fraud Detection API
Loads the trained model and exposes /predict endpoint.

Run with: uvicorn src.api:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator
from typing import Optional, List
import joblib
import numpy as np
import pandas as pd
import os

# ── Load model artifacts at startup ──────────────────────────────────────────
MODEL_DIR = os.getenv("MODEL_DIR", "model")

try:
    model          = joblib.load(os.path.join(MODEL_DIR, "xgb_model.pkl"))
    label_encoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.pkl"))
    threshold      = joblib.load(os.path.join(MODEL_DIR, "threshold.pkl"))
    metrics        = joblib.load(os.path.join(MODEL_DIR, "metrics.pkl"))
    FEATURE_NAMES  = metrics["feature_names"]
except FileNotFoundError as e:
    raise RuntimeError(f"Model not found. Run pipeline.py first. ({e})")

# ── FastAPI app ───────────────────────────────────────────────────────────────
app = FastAPI(
    title="Umba Fraud Detection API",
    description="Scores mobile-money transactions for fraud probability",
    version="1.0.0",
)


# ── Request / Response schemas ────────────────────────────────────────────────

class Transaction(BaseModel):
    TransactionID: int
    TransactionDT: int
    TransactionAmt: float = Field(..., gt=0)
    country: Optional[str] = None
    currency: Optional[str] = None
    channel: Optional[str] = None
    card_type: Optional[str] = None
    card_bank: Optional[str] = None
    card1: Optional[float] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card5: Optional[float] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    recipient_account_age_days: Optional[int] = None
    sender_prev_txn_count: Optional[int] = None
    C1: Optional[float] = None
    C2: Optional[float] = None
    C3: Optional[float] = None
    C4: Optional[float] = None
    C5: Optional[float] = None
    C6: Optional[float] = None
    C7: Optional[float] = None
    C8: Optional[float] = None
    D1: Optional[float] = None
    D2: Optional[float] = None
    D3: Optional[float] = None
    D4: Optional[float] = None
    D5: Optional[float] = None
    M1: Optional[str] = None
    M2: Optional[str] = None
    M3: Optional[str] = None
    M4: Optional[str] = None
    M5: Optional[str] = None
    M6: Optional[str] = None
    # V features
    V1:  Optional[float] = None
    V2:  Optional[float] = None
    V3:  Optional[float] = None
    V4:  Optional[float] = None
    V5:  Optional[float] = None
    V6:  Optional[float] = None
    V7:  Optional[float] = None
    V8:  Optional[float] = None
    V9:  Optional[float] = None
    V10: Optional[float] = None
    V11: Optional[float] = None
    V12: Optional[float] = None
    V13: Optional[float] = None
    V14: Optional[float] = None
    V15: Optional[float] = None
    V16: Optional[float] = None
    V17: Optional[float] = None
    V18: Optional[float] = None
    V19: Optional[float] = None
    V20: Optional[float] = None
    # Identity (optional — may not be present at score time)
    DeviceType: Optional[str] = None
    DeviceInfo: Optional[str] = None
    id_01: Optional[float] = None
    id_02: Optional[float] = None
    id_03: Optional[float] = None
    id_04: Optional[float] = None
    id_05: Optional[float] = None
    id_06: Optional[float] = None
    id_07: Optional[float] = None
    id_08: Optional[float] = None
    id_09: Optional[float] = None
    id_10: Optional[float] = None
    id_11: Optional[float] = None

    @field_validator("currency")
    @classmethod
    def currency_must_be_valid(cls, v):
        if v and v not in ("KES", "NGN"):
            raise ValueError("currency must be KES or NGN")
        return v


class PredictResponse(BaseModel):
    TransactionID: int
    fraud_probability: float
    alarm: bool
    threshold_used: float
    risk_level: str


class BatchRequest(BaseModel):
    transactions: List[Transaction]


# ── Feature engineering (mirrors pipeline.py) ────────────────────────────────

M_COLS = [f"M{i}" for i in range(1, 7)]
CAT_COLS = ["country", "currency", "channel", "card_type",
            "card_bank", "P_emaildomain", "R_emaildomain",
            "DeviceType", "DeviceInfo"]
SPARSE_COLS = ["dist1", "dist2", "D2", "D4", "D5"]

KES_MEDIAN = 5000.0   # approximate — ideally save from training
NGN_MEDIAN = 50000.0


def engineer_row(df: pd.DataFrame) -> pd.DataFrame:
    """Apply same transformations as pipeline.py to a batch DataFrame."""

    # Normalised amount
    df["amt_normalised"] = df.apply(
        lambda r: r["TransactionAmt"] / (KES_MEDIAN if r.get("currency") == "KES" else NGN_MEDIAN),
        axis=1
    )
    df["amt_log"] = np.log1p(df["TransactionAmt"])

    # Time features
    df["dt_day"]     = (df["TransactionDT"] // 86400) % 7
    df["dt_hour"]    = (df["TransactionDT"] // 3600)  % 24
    df["dt_isnight"] = ((df["dt_hour"] >= 22) | (df["dt_hour"] <= 5)).astype(int)

    # M columns
    for col in M_COLS:
        if col in df.columns:
            df[col] = df[col].map({"T": 1, "F": 0}).fillna(-1).astype(int)

    m_available = [c for c in M_COLS if c in df.columns]
    df["m_match_count"]    = (df[m_available] == 1).sum(axis=1)
    df["m_mismatch_count"] = (df[m_available] == 0).sum(axis=1)

    # Missing indicators
    for col in SPARSE_COLS:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isnull().astype(int)
        else:
            df[f"{col}_missing"] = 1  # missing entirely → treat as missing

    # Encode categoricals
    for col in CAT_COLS:
        if col not in df.columns:
            df[col] = "__missing__"
        df[col] = df[col].fillna("__missing__").astype(str)
        le = label_encoders.get(col)
        if le:
            known = set(le.classes_)
            df[col] = df[col].apply(lambda x: x if x in known else "__missing__")
            if "__missing__" not in known:
                le.classes_ = np.append(le.classes_, "__missing__")
            df[col] = le.transform(df[col])

    # Align to training feature names
    for feat in FEATURE_NAMES:
        if feat not in df.columns:
            df[feat] = 0  # fill missing features with 0

    return df[FEATURE_NAMES]


def score_risk(prob: float) -> str:
    if prob >= 0.7:   return "HIGH"
    if prob >= 0.3:   return "MEDIUM"
    return "LOW"


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": "XGBoost",
        "threshold": round(threshold, 4),
        "oof_prauc": round(metrics.get("oof_prauc", 0), 4),
    }


@app.post("/predict", response_model=PredictResponse)
def predict(txn: Transaction):
    df = pd.DataFrame([txn.model_dump()])
    try:
        X = engineer_row(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {e}")

    prob = float(model.predict_proba(X)[:, 1][0])
    return PredictResponse(
        TransactionID=txn.TransactionID,
        fraud_probability=round(prob, 6),
        alarm=prob >= threshold,
        threshold_used=round(threshold, 4),
        risk_level=score_risk(prob),
    )


@app.post("/predict/batch")
def predict_batch(batch: BatchRequest):
    df = pd.DataFrame([t.model_dump() for t in batch.transactions])
    try:
        X = engineer_row(df)
    except Exception as e:
        raise HTTPException(status_code=422, detail=f"Feature engineering failed: {e}")

    probs = model.predict_proba(X)[:, 1]
    results = []
    for i, txn in enumerate(batch.transactions):
        prob = float(probs[i])
        results.append({
            "TransactionID": txn.TransactionID,
            "fraud_probability": round(prob, 6),
            "alarm": prob >= threshold,
            "threshold_used": round(threshold, 4),
            "risk_level": score_risk(prob),
        })
    return {"predictions": results, "count": len(results)}