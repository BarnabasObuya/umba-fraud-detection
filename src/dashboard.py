"""
dashboard.py — Umba Fraud Detection Dashboard
Streamlit app for ops managers to understand model performance on test set.

Run with: streamlit run src/dashboard.py

"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Umba Fraud Detection Dashboard",
    page_icon="🔒",
    layout="wide",
)

# ── Load artifacts ────────────────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    model          = joblib.load("model/xgb_model.pkl")
    label_encoders = joblib.load("model/label_encoders.pkl")
    threshold      = joblib.load("model/threshold.pkl")
    metrics        = joblib.load("model/metrics.pkl")
    return model, label_encoders, threshold, metrics

@st.cache_data
def load_predictions():
    preds = pd.read_csv("outputs/predictions.csv")
    test  = pd.read_csv("data/test.csv")
    return preds.merge(test, on="TransactionID", how="left")

model, label_encoders, threshold, metrics = load_artifacts()
df = load_predictions()

df["alarm"] = (df["isFraud_prob"] >= threshold).astype(int)
df["risk_level"] = pd.cut(
    df["isFraud_prob"],
    bins=[-0.001, 0.3, 0.7, 1.001],
    labels=["LOW", "MEDIUM", "HIGH"]
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🔒 Umba Fraud Detection — Ops Dashboard")
st.caption(f"Model: XGBoost  |  Threshold: {threshold:.3f}  |  "
           f"OOF PR-AUC: {metrics['oof_prauc']:.4f}  |  "
           f"OOF ROC-AUC: {metrics['oof_rocauc']:.4f}")

# ── KPI Metrics ───────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
total = len(df)
flagged = df["alarm"].sum()
fraud_rate_predicted = df["isFraud_prob"].mean()

c1.metric("Total Transactions", f"{total:,}")
c2.metric("Flagged for Review", f"{flagged:,}",
          delta=f"{flagged/total*100:.1f}% of all")
c3.metric("Avg Predicted Fraud Probability", f"{fraud_rate_predicted:.3f}")
c4.metric("High-Risk Transactions", f"{(df['risk_level']=='HIGH').sum():,}")

st.divider()

# ── Score Distribution ────────────────────────────────────────────────────────
st.subheader("📊 Fraud Score Distribution")

col1, col2 = st.columns([2, 1])

with col1:
    fig = px.histogram(
        df, x="isFraud_prob", nbins=60,
        title="Distribution of Fraud Probability Scores",
        labels={"isFraud_prob": "Fraud Probability", "count": "Transactions"},
        color_discrete_sequence=["#1f77b4"]
    )
    fig.add_vline(x=threshold, line_dash="dash", line_color="red",
                  annotation_text=f"Threshold = {threshold:.3f}",
                  annotation_position="top right")
    fig.update_layout(bargap=0.05)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    risk_counts = df["risk_level"].value_counts().reset_index()
    risk_counts.columns = ["Risk Level", "Count"]
    fig2 = px.pie(
        risk_counts, values="Count", names="Risk Level",
        title="Risk Level Breakdown",
        color="Risk Level",
        color_discrete_map={"LOW": "#2ecc71", "MEDIUM": "#f39c12", "HIGH": "#e74c3c"}
    )
    st.plotly_chart(fig2, use_container_width=True)

# ── Top Flagged Transactions ──────────────────────────────────────────────────
st.subheader("🚨 Top 20 Highest-Risk Transactions")

display_cols = ["TransactionID", "isFraud_prob", "risk_level",
                "TransactionAmt", "currency", "channel", "country",
                "card_type", "P_emaildomain"]
display_cols = [c for c in display_cols if c in df.columns]

top_flagged = (
    df[display_cols]
    .sort_values("isFraud_prob", ascending=False)
    .head(20)
    .reset_index(drop=True)
)
top_flagged.index += 1

# Color-code the risk level
def highlight_risk(row):
    colors = {"HIGH": "background-color: #ffcccc",
              "MEDIUM": "background-color: #fff3cd",
              "LOW": ""}
    return [colors.get(str(row.get("risk_level", "")), "")] * len(row)

st.dataframe(
    top_flagged.style.apply(highlight_risk, axis=1),
    use_container_width=True
)

# ── Channel & Country Breakdown ───────────────────────────────────────────────
st.subheader("📍 Fraud Risk by Channel and Country")

col3, col4 = st.columns(2)

with col3:
    if "channel" in df.columns:
        ch_stats = (
            df.groupby("channel")["isFraud_prob"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "avg_fraud_prob", "count": "txn_count"})
            .sort_values("avg_fraud_prob", ascending=False)
        )
        fig3 = px.bar(
            ch_stats, x="channel", y="avg_fraud_prob",
            title="Avg Fraud Probability by Channel",
            color="avg_fraud_prob",
            color_continuous_scale="Reds",
            text=ch_stats["txn_count"].apply(lambda x: f"n={x:,}")
        )
        st.plotly_chart(fig3, use_container_width=True)

with col4:
    if "country" in df.columns:
        country_stats = (
            df.groupby("country")["isFraud_prob"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(columns={"mean": "avg_fraud_prob", "count": "txn_count"})
        )
        fig4 = px.bar(
            country_stats, x="country", y="avg_fraud_prob",
            title="Avg Fraud Probability by Country",
            color="avg_fraud_prob",
            color_continuous_scale="Oranges",
            text=country_stats["txn_count"].apply(lambda x: f"n={x:,}")
        )
        st.plotly_chart(fig4, use_container_width=True)

# ── Threshold Sensitivity ─────────────────────────────────────────────────────
st.subheader("⚙️ Threshold Sensitivity (Ops Simulator)")
st.write("Adjust the alarm threshold below to see how many transactions get flagged.")

user_threshold = st.slider(
    "Alarm Threshold", min_value=0.0, max_value=1.0,
    value=float(threshold), step=0.01
)

flagged_at_thresh = (df["isFraud_prob"] >= user_threshold).sum()
col5, col6 = st.columns(2)
col5.metric("Transactions Flagged", f"{flagged_at_thresh:,}",
            f"{flagged_at_thresh/total*100:.2f}% of all")
col6.metric("Transactions Passed", f"{total - flagged_at_thresh:,}")

# Volume at each threshold bucket
thresholds_range = np.arange(0.1, 1.0, 0.05)
flagged_counts   = [(df["isFraud_prob"] >= t).sum() for t in thresholds_range]

fig5 = go.Figure()
fig5.add_trace(go.Scatter(
    x=thresholds_range, y=flagged_counts,
    mode="lines+markers", name="Flagged Count",
    line=dict(color="red")
))
fig5.add_vline(x=user_threshold, line_dash="dash", line_color="blue",
               annotation_text="Selected threshold")
fig5.update_layout(
    title="Transactions Flagged vs Threshold",
    xaxis_title="Threshold",
    yaxis_title="# Transactions Flagged"
)
st.plotly_chart(fig5, use_container_width=True)

# ── Feature Importance ────────────────────────────────────────────────────────
st.subheader("🔍 Top Features Driving Fraud Predictions")

feat_imp = pd.DataFrame({
    "feature": metrics["feature_names"],
    "importance": model.feature_importances_
}).sort_values("importance", ascending=False).head(20)

fig6 = px.bar(
    feat_imp, x="importance", y="feature",
    orientation="h",
    title="Top 20 Feature Importances (XGBoost gain)",
    color="importance",
    color_continuous_scale="Blues"
)
fig6.update_layout(yaxis=dict(autorange="reversed"))
st.plotly_chart(fig6, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption("Umba Fraud Detection System v1.0 | Powered by XGBoost | "
           "Dashboard built by Barnabas Obuya")