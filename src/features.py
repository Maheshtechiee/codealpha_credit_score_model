"""
features.py
Domain-driven feature engineering for credit scoring.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates derived features that improve model signal.
    """
    df = df.copy()

    le = LabelEncoder()
    df["loan_purpose_enc"]  = le.fit_transform(df["loan_purpose"].fillna("personal"))
    df["payment_discipline"]= 1 / (df["num_late_payments"] + 1)
    df["income_per_year_emp"]= df["income"] / (df["employment_years"] + 1)
    df["loan_to_income"]    = df["loan_amount"] / (df["income"] + 1)
    df["debt_burden_score"] = df["debt_to_income"] * df["credit_util_ratio"]
    df["inquiry_density"]   = df["num_inquiries"] / (df["num_credit_lines"] + 1)
    df["savings_to_debt"]   = df["savings_balance"] / (df["total_debt"] + 1)
    df["risk_composite"]    = (
        df["num_late_payments"] * 0.4
        + df["credit_util_ratio"] * 0.3
        + df["debt_to_income"] * 0.2
        + df["num_inquiries"] * 0.1
    )

    df.drop(columns=["loan_purpose"], inplace=True)
    return df
