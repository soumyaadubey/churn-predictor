# Customer Churn Prediction System

![CI](https://github.com/soumyaadubey/churn-predictor/actions/workflows/ci.yml/badge.svg)

End-to-end machine learning system that predicts which customers are likely to cancel their subscription, served as a production-style REST API with automated tests and Docker deployment.

## Why this project

Churn prediction is one of the most common real-world ML problems — retaining a customer is far cheaper than acquiring a new one. This project goes beyond a notebook: it covers the full ML engineering lifecycle from data to a deployable, tested service.

## Results (held-out test set, n=1,600)

| Model | ROC-AUC | PR-AUC | F1 |
|---|---|---|---|
| **Logistic Regression (balanced)** | **0.787** | **0.651** | **0.619** |
| Gradient Boosting | 0.780 | 0.640 | 0.553 |

Class imbalance (~34% churn) is handled with class weighting, and PR-AUC is reported alongside ROC-AUC since it better reflects performance on the minority class.

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
python src/generate_data.py   # or drop a real telco CSV into data/
python src/train.py
uvicorn api.main:app --reload
```

Interactive API docs: http://localhost:8000/docs

Example request:
```bash
curl -X POST localhost:8000/predict -H "Content-Type: application/json" -d '{
  "tenure_months": 3, "monthly_charges": 95.0, "total_charges": 280.0,
  "support_calls": 4, "contract": "month-to-month",
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

- Swap the synthetic generator for the IBM Telco Churn dataset (Kaggle)
- Add SHAP explanations to the API response
- Add a GitHub Actions workflow running pytest on every push
- Track experiments with MLflow
