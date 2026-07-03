"""Train and evaluate churn models, save the best one.

Compares Logistic Regression vs Gradient Boosting with a proper
sklearn Pipeline (no leakage), stratified split, and threshold-aware
metrics (ROC-AUC, PR-AUC, F1) since churn is imbalanced.
"""
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["tenure_months", "monthly_charges", "total_charges"]
CATEGORICAL = [
    "contract",
    "internet_service",
    "payment_method",
    "paperless_billing",
    "senior_citizen",
    "tech_support",
    "online_security",
]
TARGET = "churn"


def build_pipeline(model) -> Pipeline:
    pre = ColumnTransformer(
        [
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ]
    )
    return Pipeline([("pre", pre), ("model", model)])


def main(data_path: str = "data/customers.csv") -> None:
    df = pd.read_csv(data_path)
    X, y = df.drop(columns=[TARGET]), df[TARGET]
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    candidates = {
        "logistic_regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "gradient_boosting": GradientBoostingClassifier(random_state=42),
    }

    results, best_name, best_auc, best_pipe = {}, None, -1.0, None
    for name, model in candidates.items():
        pipe = build_pipeline(model)
        pipe.fit(X_tr, y_tr)
        proba = pipe.predict_proba(X_te)[:, 1]
        preds = (proba >= 0.5).astype(int)
        metrics = {
            "roc_auc": round(roc_auc_score(y_te, proba), 4),
            "pr_auc": round(average_precision_score(y_te, proba), 4),
            "f1": round(f1_score(y_te, preds), 4),
        }
        results[name] = metrics
        print(f"{name}: {metrics}")
        if metrics["roc_auc"] > best_auc:
            best_auc, best_name, best_pipe = metrics["roc_auc"], name, pipe

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    joblib.dump(best_pipe, "models/churn_model.joblib")

    report = {
        "best_model": best_name,
        "metrics": results,
        "test_size": len(y_te),
        "churn_rate": round(float(y.mean()), 4),
    }
    Path("reports/metrics.json").write_text(json.dumps(report, indent=2))
    print(f"\nBest model: {best_name} (ROC-AUC {best_auc})")
    print(classification_report(y_te, best_pipe.predict(X_te), digits=3))


if __name__ == "__main__":
    main()
