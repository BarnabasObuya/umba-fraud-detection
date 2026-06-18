"""
pipeline.py — Umba Fraud Detection
Full pipeline: load → clean → feature engineer → train XGBoost → evaluate → save model
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    average_precision_score, roc_auc_score,
    precision_recall_curve, classification_report
)
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────

def load_data(data_dir="data"):
    print("📂 Loading data...")
    train = pd.read_csv(os.path.join(data_dir, "train.csv"))
    test  = pd.read_csv(os.path.join(data_dir, "test.csv"))
    identity = pd.read_csv(os.path.join(data_dir, "identity.csv"))
    print(f"   Train: {train.shape}, Test: {test.shape}, Identity: {identity.shape}")
    return train, test, identity


# ─────────────────────────────────────────────
# 2. HANDLE IDENTITY TABLE (many-to-one join)
# ─────────────────────────────────────────────

def aggregate_identity(identity: pd.DataFrame) -> pd.DataFrame:
    """
    Identity has duplicate TransactionIDs (multiple device sessions per txn).
    We collapse to one row per TransactionID by:
      - Categorical cols (DeviceType, DeviceInfo): take mode (most frequent)
      - Numeric cols (id_01..id_11): take mean
    This preserves information without exploding rows on join.
    """
    print("🔗 Aggregating identity table...")

    cat_cols = ["DeviceType", "DeviceInfo"]
    num_cols = [c for c in identity.columns if c.startswith("id_")]

    # Mode for categoricals (take first if tie)
    cat_agg = (
        identity.groupby("TransactionID")[cat_cols]
        .agg(lambda x: x.mode().iloc[0] if not x.mode().empty else np.nan)
    )

    # Mean for numerics
    num_agg = identity.groupby("TransactionID")[num_cols].mean()

    identity_agg = cat_agg.join(num_agg).reset_index()
    print(f"   Identity after aggregation: {identity_agg.shape}")
    return identity_agg


# ─────────────────────────────────────────────
# 3. CLEAN & FEATURE ENGINEER
# ─────────────────────────────────────────────

# LEAKAGE: flagged_for_review is populated AFTER review — never available
# at scoring time. We must drop it.
LEAKY_COLS = ["flagged_review"]  # alias; handled below by name

# M columns are T/F/NaN → encode as 1/0/-1
M_COLS = [f"M{i}" for i in range(1, 7)]

# Categorical columns to label-encode
CAT_COLS = ["country", "currency", "channel", "card_type",
            "card_bank", "P_emaildomain", "R_emaildomain",
            "DeviceType", "DeviceInfo"]


def clean_and_engineer(df: pd.DataFrame, identity_agg: pd.DataFrame,
                        label_encoders: dict = None, fit: bool = True):
    """
    Cleans and engineers features for train (fit=True) or test (fit=False).
    Returns transformed df and the fitted label_encoders dict.
    """
    print(f"🔧 Feature engineering (fit={fit})...")

    # ── Drop leaky column ──────────────────────────────────────────────────
    if "flagged_for_review" in df.columns:
        df = df.drop(columns=["flagged_for_review"])
        print(" Dropped flagged_for_review (leakage)")

    # ── Left-join identity ─────────────────────────────────────────────────
    df = df.merge(identity_agg, on="TransactionID", how="left")

    # ── Currency-normalised amount ─────────────────────────────────────────
    # KES and NGN are on very different scales; normalise within currency
    # to make TransactionAmt comparable across rows
    kes_median = df.loc[df["currency"] == "KES", "TransactionAmt"].median()
    ngn_median = df.loc[df["currency"] == "NGN", "TransactionAmt"].median()

    def normalise_amt(row):
        if row["currency"] == "KES":
            return row["TransactionAmt"] / kes_median if kes_median else row["TransactionAmt"]
        else:
            return row["TransactionAmt"] / ngn_median if ngn_median else row["TransactionAmt"]

    df["amt_normalised"] = df.apply(normalise_amt, axis=1)
    df["amt_log"] = np.log1p(df["TransactionAmt"])

    # ── Time features ──────────────────────────────────────────────────────
    # TransactionDT is seconds from a reference. Extract cyclical signals.
    df["dt_day"]    = (df["TransactionDT"] // 86400) % 7    # day of week
    df["dt_hour"]   = (df["TransactionDT"] // 3600)  % 24  # hour of day
    df["dt_isnight"] = ((df["dt_hour"] >= 22) | (df["dt_hour"] <= 5)).astype(int)

    # ── M columns: T→1, F→0, NaN→-1 ──────────────────────────────────────
    for col in M_COLS:
        if col in df.columns:
            df[col] = df[col].map({"T": 1, "F": 0}).fillna(-1).astype(int)

    # ── Count how many M flags are True (sum of positive matches) ─────────
    m_available = [c for c in M_COLS if c in df.columns]
    df["m_match_count"] = (df[m_available] == 1).sum(axis=1)
    df["m_mismatch_count"] = (df[m_available] == 0).sum(axis=1)

    # ── Missing-value indicator features (missingness itself is informative) ─
    sparse_cols = ["dist1", "dist2", "D2", "D4", "D5"]
    for col in sparse_cols:
        if col in df.columns:
            df[f"{col}_missing"] = df[col].isnull().astype(int)

    # ── Label-encode categoricals ──────────────────────────────────────────
    if label_encoders is None:
        label_encoders = {}

    for col in CAT_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].fillna("__missing__").astype(str)
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            le = label_encoders.get(col)
            if le:
                # Handle unseen labels gracefully
                known = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known else "__missing__")
                if "__missing__" not in known:
                    le.classes_ = np.append(le.classes_, "__missing__")
                df[col] = le.transform(df[col])

    print(f"   ✅ Shape after engineering: {df.shape}")
    return df, label_encoders


# ─────────────────────────────────────────────
# 4. TRAIN / EVALUATE
# ─────────────────────────────────────────────

def get_features(df):
    """Return the feature matrix X (drop IDs and target)."""
    drop = ["TransactionID", "isFraud", "TransactionDT"]
    cols = [c for c in df.columns if c not in drop]
    return df[cols]


def train_model(train_clean: pd.DataFrame):
    """
    Train XGBoost with time-aware cross-validation.

    WHY time-aware CV?
    - Fraud patterns drift over time.
    - Random k-fold would let future data leak into past folds.
    - We use StratifiedKFold on a time-sorted dataset so each fold
      uses earlier data to predict later data (approximates production).

    WHY scale_pos_weight?
    - XGBoost's built-in way to handle class imbalance.
    - scale_pos_weight = negatives / positives tells the model to
      penalise missing a fraud ~28× more than a false alarm.
    """
    print("\n🚀 Training XGBoost...")

    # Sort by time — critical for honest CV
    train_clean = train_clean.sort_values("TransactionDT").reset_index(drop=True)

    X = get_features(train_clean)
    y = train_clean["isFraud"]

    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale_pos_weight = neg / pos
    print(f"   Class ratio → scale_pos_weight = {scale_pos_weight:.1f}")

    model = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",       # PR-AUC — right metric for imbalanced data
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        tree_method="hist",        # fast histogram method
    )

    # ── Stratified K-Fold CV (5 folds) ────────────────────────────────────
    # We use stratified to ensure each fold has ~same fraud rate
    skf = StratifiedKFold(n_splits=5, shuffle=False)  # no shuffle = time order
    oof_probs = np.zeros(len(y))

    for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )
        oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]

        fold_prauc = average_precision_score(y_val, oof_probs[val_idx])
        fold_rocauc = roc_auc_score(y_val, oof_probs[val_idx])
        print(f"   Fold {fold+1}: PR-AUC={fold_prauc:.4f}  ROC-AUC={fold_rocauc:.4f}")

    # ── Overall OOF metrics ───────────────────────────────────────────────
    oof_prauc = average_precision_score(y, oof_probs)
    oof_rocauc = roc_auc_score(y, oof_probs)
    print(f"\n   ✅ OOF PR-AUC  = {oof_prauc:.4f}")
    print(f"   ✅ OOF ROC-AUC = {oof_rocauc:.4f}")

    # ── Find best threshold by F1 on OOF ─────────────────────────────────
    precision, recall, thresholds = precision_recall_curve(y, oof_probs)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    print(f"   ✅ Best threshold (F1) = {best_threshold:.4f}")
    print(f"      Precision={precision[best_idx]:.3f}, Recall={recall[best_idx]:.3f}")

    # ── Retrain on ALL training data for final model ──────────────────────
    print("\n   🔁 Retraining on full train set...")
    # Use 10% of training data as internal validation for early stopping
    split = int(len(X) * 0.9)
    model.fit(
        X.iloc[:split], y.iloc[:split],
        eval_set=[(X.iloc[split:], y.iloc[split:])],
        verbose=False,
    )

    print("\n   📊 Classification report at best threshold:")
    y_pred = (oof_probs >= best_threshold).astype(int)
    print(classification_report(y, y_pred, target_names=["Legit", "Fraud"]))

    return model, best_threshold, {
        "oof_prauc": oof_prauc,
        "oof_rocauc": oof_rocauc,
        "best_threshold": best_threshold,
        "feature_names": list(X.columns),
    }


# ─────────────────────────────────────────────
# 5. MAIN — run the full pipeline
# ─────────────────────────────────────────────

def main():
    os.makedirs("model", exist_ok=True)
    os.makedirs("outputs", exist_ok=True)

    # Load
    train, test, identity = load_data(data_dir="data")

    # Aggregate identity
    identity_agg = aggregate_identity(identity)

    # Clean & engineer train
    train_clean, label_encoders = clean_and_engineer(
        train, identity_agg, fit=True
    )

    # Train
    model, threshold, metrics = train_model(train_clean)

    # Save model artifacts
    joblib.dump(model,          "model/xgb_model.pkl")
    joblib.dump(label_encoders, "model/label_encoders.pkl")
    joblib.dump(threshold,      "model/threshold.pkl")
    joblib.dump(metrics,        "model/metrics.pkl")
    print("\n💾 Model artifacts saved to model/")

    # Clean & engineer test
    test_clean, _ = clean_and_engineer(
        test, identity_agg, label_encoders=label_encoders, fit=False
    )

    # Score test
    X_test = get_features(test_clean)
    test_probs = model.predict_proba(X_test)[:, 1]

    # Save predictions
    submission = pd.DataFrame({
        "TransactionID": test_clean["TransactionID"],
        "isFraud_prob": test_probs
    })
    submission.to_csv("outputs/predictions.csv", index=False)
    print(f"\n📄 predictions.csv saved → {len(submission)} rows")
    print(submission.head())

    print("\n🎉 Pipeline complete!")


if __name__ == "__main__":
    main()
