import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
from fastapi.testclient import TestClient

from src.generate_data import generate


def test_data_generation_shape_and_target():
    df = generate(500)
    assert len(df) == 500
    assert df["churn"].isin([0, 1]).all()
    assert 0.05 < df["churn"].mean() < 0.6  # sane imbalance


def test_no_missing_values():
    assert generate(300).isna().sum().sum() == 0


def test_api_predict_and_validation():
    from api.main import app

    client = TestClient(app)
    assert client.get("/health").json() == {"status": "ok"}

    payload = {
        "tenure_months": 3,
        "monthly_charges": 95.0,
        "total_charges": 280.0,
        "support_calls": 4,
        "contract": "month-to-month",
        "internet_service": "fiber",
        "payment_method": "electronic_check",
        "paperless_billing": 1,
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    body = res.json()
    assert 0.0 <= body["churn_probability"] <= 1.0
    assert body["risk_band"] in {"low", "medium", "high"}

    # invalid contract value should be rejected
    bad = {**payload, "contract": "lifetime"}
    assert client.post("/predict", json=bad).status_code == 422
