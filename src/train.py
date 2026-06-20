"""
train.py
Train Logistic Regression, Decision Tree, and Random Forest models.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier


def train_models(X_train, y_train) -> dict:
    """
    Train all three models with 5-fold CV reporting.
    Returns a dict of fitted models.
    """
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000, random_state=42, class_weight="balanced"
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=6, min_samples_leaf=20, random_state=42, class_weight="balanced"
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_leaf=10,
            random_state=42, class_weight="balanced", n_jobs=-1
        ),
    }

    print("\n" + "═"*60)
    print("  TRAINING MODELS")
    print("═"*60)
    for name, model in models.items():
        model.fit(X_train, y_train)
        cv = cross_val_score(model, X_train, y_train, cv=5, scoring="roc_auc")
        print(f"  ✅ {name:<25} | 5-Fold CV ROC-AUC: {cv.mean():.4f} ± {cv.std():.4f}")

    return models
