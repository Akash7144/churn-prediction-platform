# Customer Churn Prediction Platform

End-to-end ML project: data cleaning → feature engineering → model training →
explainability (SHAP) → API → deployment.

## Dataset
IBM/Kaggle Telco Customer Churn — 7,043 customers, 21 features.
Source: https://www.kaggle.com/blastchar/telco-customer-churn

## Project structure
```
data/        raw + cleaned CSVs
notebooks/   EDA plots and exploration
src/         reusable pipeline code (data prep, training, explainability)
models/      saved model artifacts
api/         FastAPI serving layer
tests/       unit tests
```

## Progress log

### Stage 1 — EDA & Data Cleaning ✅
- Found and fixed a known data quality issue: 11 customers with `tenure == 0`
  had blank `TotalCharges` (they're new sign-ups, not yet billed). Converted
  and filled with 0 rather than dropping the rows.
- Confirmed target class imbalance: ~73% No churn / ~27% Yes churn.
  This rules out accuracy as the primary metric — using precision/recall/F1/AUC
  instead.
- Initial signal: churn is concentrated in month-to-month contracts,
  fiber-optic internet customers, low-tenure customers, and higher
  monthly-charge customers.

### Stage 2 — Feature engineering + baseline model (next)
### Stage 3 — XGBoost/LightGBM + class imbalance handling
### Stage 4 — Hyperparameter tuning (Optuna)
### Stage 5 — SHAP explainability
### Stage 6 — FastAPI serving
### Stage 7 — Docker + deployment
### Stage 8 — Streamlit dashboard

## How to run (Stage 1)
```bash
pip install -r requirements.txt
python src/eda.py
```
