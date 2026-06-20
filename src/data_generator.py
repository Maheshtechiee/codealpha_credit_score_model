"""
data_generator.py
Generates a realistic synthetic credit dataset with domain-driven features.
"""

import numpy as np
import pandas as pd


def generate_credit_dataset(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic credit dataset.
    Features mirror real-world bureau data: income, debts,
    payment history, loan amounts, utilization, etc.
    """
    rng = np.random.default_rng(random_state)

    income              = rng.normal(50000, 20000, n_samples).clip(10000, 200000)
    age                 = rng.integers(21, 70, n_samples).astype(float)
    employment_years    = rng.integers(0, 30, n_samples).astype(float)
    num_credit_lines    = rng.integers(1, 15, n_samples).astype(float)
    credit_util_ratio   = rng.uniform(0.0, 1.0, n_samples)
    num_late_payments   = rng.integers(0, 10, n_samples).astype(float)
    total_debt          = rng.normal(30000, 15000, n_samples).clip(0, 150000)
    debt_to_income      = (total_debt / income).clip(0, 5)
    loan_amount         = rng.normal(15000, 8000, n_samples).clip(1000, 80000)
    num_inquiries       = rng.integers(0, 8, n_samples).astype(float)
    loan_purpose        = rng.choice(["home", "car", "education", "personal", "business"], n_samples)
    has_mortgage        = rng.integers(0, 2, n_samples).astype(float)
    savings_balance     = rng.normal(10000, 8000, n_samples).clip(0, 100000)

    score = (
        (income / 200000) * 30
        + (age / 70) * 10
        + (employment_years / 30) * 10
        - credit_util_ratio * 20
        - (num_late_payments / 10) * 25
        - debt_to_income * 10
        - (num_inquiries / 8) * 10
        + (savings_balance / 100000) * 10
        + (has_mortgage) * 5
        + rng.uniform(-5, 5, n_samples)
    )
    threshold    = np.percentile(score, 40)
    creditworthy = (score >= threshold).astype(int)

    df = pd.DataFrame({
        "age":               age,
        "income":            income.round(2),
        "employment_years":  employment_years,
        "num_credit_lines":  num_credit_lines,
        "credit_util_ratio": credit_util_ratio.round(4),
        "num_late_payments": num_late_payments,
        "total_debt":        total_debt.round(2),
        "debt_to_income":    debt_to_income.round(4),
        "loan_amount":       loan_amount.round(2),
        "num_inquiries":     num_inquiries,
        "loan_purpose":      loan_purpose,
        "has_mortgage":      has_mortgage,
        "savings_balance":   savings_balance.round(2),
        "creditworthy":      creditworthy,
    })

    for col in ["income", "employment_years", "savings_balance", "credit_util_ratio"]:
        mask = rng.random(n_samples) < 0.03
        df.loc[mask, col] = np.nan

    return df
