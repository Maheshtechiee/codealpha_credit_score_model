"""
evaluate.py
Evaluate trained models and produce a summary DataFrame.
"""

import pandas as pd
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score,
    precision_score, recall_score, roc_auc_score,
)


def evaluate_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Print full classification reports and return a summary DataFrame.
    """
    print("\n" + "═"*60)
    print("  EVALUATION RESULTS")
    print("═"*60)

    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec  = recall_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred)
        auc  = roc_auc_score(y_test, y_prob)

        print(f"\n  ── {name} ──")
        print(f"  Accuracy : {acc:.4f}  |  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}  |  F1-Score : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print(classification_report(y_test, y_pred, target_names=["Default (0)", "Good Credit (1)"]))

        results.append({
            "Model":     name,
            "Accuracy":  round(acc,  4),
            "Precision": round(prec, 4),
            "Recall":    round(rec,  4),
            "F1-Score":  round(f1,   4),
            "ROC-AUC":   round(auc,  4),
        })

    return pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
