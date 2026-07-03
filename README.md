# Customer Churn Prediction System

![CI](https://github.com/YOUR_USERNAME/churn-predictor/actions/workflows/ci.yml/badge.svg)

End-to-end machine learning system that predicts which customers are likely to cancel their subscription, served as a production-style REST API with automated tests and Docker deployment.

## Why this project

Churn prediction is one of the most common real-world ML problems — retaining a customer is far cheaper than acquiring a new one. This project goes beyond a notebook: it covers the full ML engineering lifecycle from data to a deployable, tested service.

## Results — real IBM Telco dataset (held-out test set, n=1,409)

| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| **Gradient Boosting** | **0.843** | **0.661** | **0.593** |
| Logistic Regression (balanced) | 0.840 | 0.636 | 0.617 |

Trained on the IBM Telco Customer Churn dataset (7,043 real customers, ~27% churn). PR-AUC is reported alongside ROC-AUC since it better reflects performance on the imbalanced minority class.

## Architecture

```
data → sklearn Pipeline (scaling + one-hot encoding) → model comparison
     → best model serialized (joblib) → FastAPI /predict endpoint
     → Pydantic input validation → risk banding (low/medium/high)
```

Key engineering decisions:
- **Leakage-free preprocessing** — all transforms live inside a sklearn `Pipeline`, fit only on training data
- **Stratified split** — preserves churn rate across train/test
- **Schema validation** — Pydantic rejects malformed inputs with 422 before they reach the model
- **Reproducibility** — fixed seeds, pinned dependencies, metrics written to `reports/metrics.json`

## Quickstart

```bash
pip install -r requirements.txt

# Real data (recommended):
curl -sL -o data/telco_raw.csv https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
python src/prepare_telco.py data/telco_raw.csv

# Or synthetic fallback (used by CI):
# python src/generate_data.py

python src/train.py
uvicorn api.main:app --reload
```

Interactive API docs: http://localhost:8000/docs

Example request:
```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "tenure_months": 3, "monthly_charges": 95.0, "total_charges": 280.0,
  "senior_citizen": 0,\n  "tech_support": "no", "online_security": "no", "contract": "month-to-month",
  "internet_service": "fiber", "payment_method": "electronic_check",
  "paperless_billing": 1
}'
# → {"churn_probability": 0.87, "risk_band": "high"}
```

## Tests

```bash
pytest tests/ -q   # data integrity, prediction contract, input validation
```

## Docker

```bash
docker build -t churn-api .
docker run -p 8000:8000 churn-api
```

## Extending it

- Add SHAP explanations to the API response
- Add a GitHub Actions workflow running pytest on every push
- Track experiments with MLflow
