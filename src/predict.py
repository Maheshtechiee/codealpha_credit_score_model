"""
predict.py
Demo prediction for a single new loan applicant.
"""

import pandas as pd


def predict_applicant(model, imputer, scaler, feature_names: list):
    """Predict creditworthiness for a sample new applicant."""
    sample = {
        "age": 35, "income": 62000, "employment_years": 8,
        "num_credit_lines": 5, "credit_util_ratio": 0.35,
        "num_late_payments": 1, "total_debt": 22000,
        "debt_to_income": 0.35, "loan_amount": 18000,
        "num_inquiries": 2, "loan_purpose_enc": 2,
        "has_mortgage": 0, "savings_balance": 8000,
        "payment_discipline":   1 / (1+1),
        "income_per_year_emp":  62000 / (8+1),
        "loan_to_income":       18000 / (62000+1),
        "debt_burden_score":    0.35 * 0.35,
        "inquiry_density":      2 / (5+1),
        "savings_to_debt":      8000 / (22000+1),
        "risk_composite":       1*0.4 + 0.35*0.3 + 0.35*0.2 + 2*0.1,
    }

    df    = pd.DataFrame([sample])[feature_names]
    X_imp = imputer.transform(df)
    X_sc  = scaler.transform(X_imp)
    prob  = model.predict_proba(X_sc)[0][1]
    label = "✅ CREDITWORTHY" if prob >= 0.5 else "❌ HIGH RISK"

    print("\n" + "═"*60)
    print("  NEW APPLICANT PREDICTION (Random Forest)")
    print("═"*60)
    print(f"  Income          : ₹{sample['income']:,.0f}")
    print(f"  Debt-to-Income  : {sample['debt_to_income']:.2f}")
    print(f"  Late Payments   : {sample['num_late_payments']}")
    print(f"  Credit Util.    : {sample['credit_util_ratio']:.0%}")
    print(f"  Employment Yrs  : {sample['employment_years']}")
    print(f"  Loan Amount     : ₹{sample['loan_amount']:,.0f}")
    print()
    print(f"  → Probability   : {prob:.4f} ({prob*100:.1f}%)")
    print(f"  → Decision      : {label}")
    print("═"*60)
