"""REST API serving the trained churn model.

Run: uvicorn api.main:app --reload
Docs auto-generated at http://localhost:8000/docs
"""
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

app = FastAPI(title="Churn Prediction API", version="1.0.0")
MODEL = joblib.load("models/churn_model.joblib")

RISK_BANDS = [(0.7, "high"), (0.4, "medium"), (0.0, "low")]


class Customer(BaseModel):
    tenure_months: int = Field(ge=0, le=100)
    monthly_charges: float = Field(gt=0)
    total_charges: float = Field(ge=0)
    support_calls: int = Field(ge=0)
    contract: Literal["month-to-month", "one_year", "two_year"]
    internet_service: Literal["dsl", "fiber", "none"]
    payment_method: Literal[
        "electronic_check", "mailed_check", "bank_transfer", "credit_card"
    ]
    paperless_billing: Literal[0, 1]


class Prediction(BaseModel):
    churn_probability: float
    risk_band: str


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/predict", response_model=Prediction)
def predict(customer: Customer) -> Prediction:
    row = pd.DataFrame([customer.model_dump()])
    proba = float(MODEL.predict_proba(row)[0, 1])
    band = next(label for cutoff, label in RISK_BANDS if proba >= cutoff)
    return Prediction(churn_probability=round(proba, 4), risk_band=band)
