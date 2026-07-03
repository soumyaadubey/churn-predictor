"""Generate a realistic synthetic telco customer dataset.

Used for demos/CI. For real results, drop the IBM Telco Churn CSV
into data/ and point the pipeline at it instead.
"""
import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)


def generate(n: int = 8000) -> pd.DataFrame:
    tenure = RNG.integers(0, 72, n)
    monthly = np.round(RNG.uniform(18, 118, n), 2)
    contract = RNG.choice(
        ["month-to-month", "one_year", "two_year"], n, p=[0.55, 0.25, 0.20]
    )
    internet = RNG.choice(["dsl", "fiber", "none"], n, p=[0.35, 0.45, 0.20])
    support_calls = RNG.poisson(1.2, n)
    payment = RNG.choice(
        ["electronic_check", "mailed_check", "bank_transfer", "credit_card"], n
    )
    paperless = RNG.integers(0, 2, n)

    # Churn probability driven by realistic factors
    logit = (
        -2.2
        - 0.035 * tenure
        + 0.018 * monthly
        + 0.45 * support_calls
        + np.where(contract == "month-to-month", 1.1, 0)
        + np.where(contract == "two_year", -0.9, 0)
        + np.where(internet == "fiber", 0.5, 0)
        + np.where(payment == "electronic_check", 0.4, 0)
        + RNG.normal(0, 0.5, n)
    )
    churn = (RNG.uniform(0, 1, n) < 1 / (1 + np.exp(-logit))).astype(int)

    return pd.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charges": monthly,
            "total_charges": np.round(tenure * monthly * RNG.uniform(0.9, 1.1, n), 2),
            "contract": contract,
            "internet_service": internet,
            "support_calls": support_calls,
            "payment_method": payment,
            "paperless_billing": paperless,
            "churn": churn,
        }
    )


if __name__ == "__main__":
    df = generate()
    df.to_csv("data/customers.csv", index=False)
    print(f"Wrote {len(df)} rows, churn rate = {df.churn.mean():.1%}")
