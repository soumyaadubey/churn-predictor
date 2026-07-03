"""Convert the raw IBM Telco Churn CSV into the pipeline schema.

Source: https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
Usage:  python src/prepare_telco.py data/telco_raw.csv
"""
import sys

import pandas as pd

CONTRACT_MAP = {"Month-to-month": "month-to-month", "One year": "one_year", "Two year": "two_year"}
INTERNET_MAP = {"DSL": "dsl", "Fiber optic": "fiber", "No": "none"}
PAYMENT_MAP = {
    "Electronic check": "electronic_check",
    "Mailed check": "mailed_check",
    "Bank transfer (automatic)": "bank_transfer",
    "Credit card (automatic)": "credit_card",
}
SERVICE_MAP = {"Yes": "yes", "No": "no", "No internet service": "no_internet"}


def prepare(raw_path: str, out_path: str = "data/customers.csv") -> pd.DataFrame:
    raw = pd.read_csv(raw_path)

    # TotalCharges has blank strings for brand-new customers (tenure=0)
    total = pd.to_numeric(raw["TotalCharges"], errors="coerce").fillna(0.0)

    df = pd.DataFrame(
        {
            "tenure_months": raw["tenure"].astype(int),
            "monthly_charges": raw["MonthlyCharges"].astype(float),
            "total_charges": total,
            "senior_citizen": raw["SeniorCitizen"].astype(int),
            "contract": raw["Contract"].map(CONTRACT_MAP),
            "internet_service": raw["InternetService"].map(INTERNET_MAP),
            "payment_method": raw["PaymentMethod"].map(PAYMENT_MAP),
            "paperless_billing": (raw["PaperlessBilling"] == "Yes").astype(int),
            "tech_support": raw["TechSupport"].map(SERVICE_MAP),
            "online_security": raw["OnlineSecurity"].map(SERVICE_MAP),
            "churn": (raw["Churn"] == "Yes").astype(int),
        }
    )

    assert df.isna().sum().sum() == 0, "Unmapped values found in raw data"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}, churn rate = {df.churn.mean():.1%}")
    return df


if __name__ == "__main__":
    raw = sys.argv[1] if len(sys.argv) > 1 else "data/telco_raw.csv"
    prepare(raw)
