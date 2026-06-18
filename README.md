# Umba Fraud Detection — v1

## How to Run Everything

### 1. Setup
```bash
pip install -r requirements.txt
```
Place the four CSVs in `data/`: `train.csv`, `test.csv`, `identity.csv`, `sample_submission.csv`.

### 2. Run the ML Pipeline
```bash
python src/pipeline.py
```
This produces:
- `model/xgb_model.pkl` — trained XGBoost model
- `model/label_encoders.pkl`, `model/threshold.pkl`, `model/metrics.pkl`
- `outputs/predictions.csv` — scores for test.csv

### 3. Start the API
```bash
uvicorn src.api:app --reload --port 8000
```
- Health check: `GET http://localhost:8000/health`
- Score a transaction: `POST http://localhost:8000/predict`
- Batch scoring: `POST http://localhost:8000/predict/batch`

### 4. Launch Dashboard
```bash
streamlit run src/dashboard.py
```
Opens at `http://localhost:8501`

---

## My Approach

### Key data integrity findings
1. **`flagged_for_review` is a data leak** — populated post-review, not available at score time. Dropped immediately.
2. **Identity table has duplicate TransactionIDs** — aggregated with mean/mode before joining (left join: ~26% of train has identity data).
3. **Two currencies (KES/NGN)** — raw `TransactionAmt` is not comparable across countries; log-transformed and median-normalised per currency.
4. **Time ordering** — train DT ends before test DT begins; used StratifiedKFold without shuffle to respect temporal ordering in CV.

### Model choices
- **XGBoost** with `scale_pos_weight = neg/pos ≈ 28` to handle 3.4% fraud rate.
- **Primary metric: PR-AUC** — the right metric under heavy class imbalance.
- **Threshold tuning** by maximising F1 on OOF predictions.
- **Missing values** handled natively by XGBoost (no imputation needed).

### What I'd do with more time
- Frequency encoding for high-cardinality categoricals (card_bank, email domains)
- Time-based aggregation features: fraud rate per card in last 24h, velocity counts
- Calibration (Platt scaling / isotonic) to make probabilities trustworthy
- Proper walk-forward CV (time-series split) instead of stratified k-fold
- SHAP values for per-transaction explainability in the dashboard
- Docker Compose for one-command deployment

### Monitoring in production
- Track score distribution daily; alert if it shifts (PSI > 0.2)
- Log fraud rate among alarmed transactions weekly
- Retrain monthly as confirmed fraud labels accumulate from chargebacks
- Feature drift monitoring per column (KS test)