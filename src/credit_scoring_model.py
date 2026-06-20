# ============================================================
#   CREDIT SCORING MODEL — Mahesh | AI/ML Portfolio Project
#   Task 1: Predict creditworthiness using financial history
#   Models: Logistic Regression | Decision Tree | Random Forest
# ============================================================

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve,
    f1_score, precision_score, recall_score, accuracy_score
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
import seaborn as sns
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  STEP 1 — SYNTHETIC DATASET GENERATION
# ─────────────────────────────────────────────


def generate_credit_dataset(n_samples: int = 2000, random_state: int = 42) -> pd.DataFrame:
    """
    Generates a realistic synthetic credit dataset.
    Features mirror real-world bureau data: income, debts,
    payment history, loan amounts, utilization, etc.
    """
    rng = np.random.default_rng(random_state)

    income = rng.normal(50000, 20000, n_samples).clip(10000, 200000)
    age = rng.integers(21, 70, n_samples).astype(float)
    employment_years = rng.integers(0, 30, n_samples).astype(float)
    num_credit_lines = rng.integers(1, 15, n_samples).astype(float)
    # 0 = low usage, 1 = maxed out
    credit_util_ratio = rng.uniform(0.0, 1.0, n_samples)
    num_late_payments = rng.integers(0, 10, n_samples).astype(float)
    total_debt = rng.normal(30000, 15000, n_samples).clip(0, 150000)
    debt_to_income = (total_debt / income).clip(0, 5)
    loan_amount = rng.normal(15000, 8000, n_samples).clip(1000, 80000)
    num_inquiries = rng.integers(0, 8, n_samples).astype(float)
    loan_purpose = rng.choice(
        ["home", "car", "education", "personal", "business"], n_samples)
    has_mortgage = rng.integers(0, 2, n_samples).astype(float)
    savings_balance = rng.normal(10000, 8000, n_samples).clip(0, 100000)

    # ---------- Creditworthiness logic (domain-realistic) ----------
    # Higher score → more likely creditworthy
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
        + rng.uniform(-5, 5, n_samples)   # noise
    )
    threshold = np.percentile(score, 40)   # ~40% default rate (realistic)
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
        "creditworthy":      creditworthy          # 1 = Good, 0 = Default/Bad
    })

    # Inject ~3% missing values (realistic bureau gaps)
    for col in ["income", "employment_years", "savings_balance", "credit_util_ratio"]:
        mask = rng.random(n_samples) < 0.03
        df.loc[mask, col] = np.nan

    return df


# ─────────────────────────────────────────────
#  STEP 2 — FEATURE ENGINEERING
# ─────────────────────────────────────────────

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates domain-driven derived features that improve model signal.
    """
    df = df.copy()

    # Encode categorical
    le = LabelEncoder()
    df["loan_purpose_enc"] = le.fit_transform(
        df["loan_purpose"].fillna("personal"))

    # Derived features
    df["payment_discipline"] = 1 / \
        (df["num_late_payments"] + 1)   # inverse of late payments
    df["income_per_year_emp"] = df["income"] / (df["employment_years"] + 1)
    df["loan_to_income"] = df["loan_amount"] / (df["income"] + 1)
    df["debt_burden_score"] = df["debt_to_income"] * df["credit_util_ratio"]
    df["inquiry_density"] = df["num_inquiries"] / (df["num_credit_lines"] + 1)
    df["savings_to_debt"] = df["savings_balance"] / (df["total_debt"] + 1)
    df["risk_composite"] = (
        df["num_late_payments"] * 0.4 +
        df["credit_util_ratio"] * 0.3 +
        df["debt_to_income"] * 0.2 +
        df["num_inquiries"] * 0.1
    )

    df.drop(columns=["loan_purpose"], inplace=True)
    return df


# ─────────────────────────────────────────────
#  STEP 3 — PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df: pd.DataFrame):
    """
    Impute missing values, split features / target, scale numerics.
    Returns: X_train, X_test, y_train, y_test, feature_names
    """
    target = "creditworthy"
    X = df.drop(columns=[target])
    y = df[target]

    feature_names = X.columns.tolist()

    # Impute → Scale
    imputer = SimpleImputer(strategy="median")
    X_imputed = imputer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_imputed, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    return X_train_sc, X_test_sc, X_train, X_test, y_train, y_test, feature_names, imputer, scaler


# ─────────────────────────────────────────────
#  STEP 4 — TRAIN MODELS
# ─────────────────────────────────────────────

def train_models(X_train, y_train):
    """
    Train Logistic Regression, Decision Tree, Random Forest.
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
        cv_scores = cross_val_score(
            model, X_train, y_train, cv=5, scoring="roc_auc")
        print(
            f"  ✅ {name:<25} | 5-Fold CV ROC-AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

    return models


# ─────────────────────────────────────────────
#  STEP 5 — EVALUATE MODELS
# ─────────────────────────────────────────────

def evaluate_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Evaluate all models and print full classification reports.
    Returns a summary DataFrame.
    """
    print("\n" + "═"*60)
    print("  EVALUATION RESULTS")
    print("═"*60)

    results = []
    for name, model in models.items():
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)

        print(f"\n  ── {name} ──")
        print(f"  Accuracy : {acc:.4f}  |  Precision: {prec:.4f}")
        print(f"  Recall   : {rec:.4f}  |  F1-Score : {f1:.4f}")
        print(f"  ROC-AUC  : {auc:.4f}")
        print()
        print(classification_report(y_test, y_pred,
                                    target_names=["Default (0)", "Good Credit (1)"]))

        results.append({
            "Model":     name,
            "Accuracy":  round(acc,  4),
            "Precision": round(prec, 4),
            "Recall":    round(rec,  4),
            "F1-Score":  round(f1,   4),
            "ROC-AUC":   round(auc,  4),
        })

    return pd.DataFrame(results).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)


# ─────────────────────────────────────────────
#  STEP 6 — VISUALIZATIONS (6-panel dashboard)
# ─────────────────────────────────────────────

PALETTE = {
    "Logistic Regression": "#4F8EF7",
    "Decision Tree":       "#F7A24F",
    "Random Forest":       "#4FD18A",
}


def plot_dashboard(models, X_test, y_test, feature_names, summary_df, df_raw):
    """
    6-panel diagnostic dashboard saved as PNG.
    """
    fig = plt.figure(figsize=(20, 18), facecolor="#0F1117")
    fig.suptitle("CREDIT SCORING MODEL — DIAGNOSTIC DASHBOARD",
                 fontsize=18, fontweight="bold", color="white", y=0.98)

    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    axes = {
        "roc":        fig.add_subplot(gs[0, 0]),
        "pr":         fig.add_subplot(gs[0, 1]),
        "cm":         fig.add_subplot(gs[0, 2]),
        "metrics":    fig.add_subplot(gs[1, 0:2]),
        "feat_imp":   fig.add_subplot(gs[1, 2]),
        "dist":       fig.add_subplot(gs[2, 0]),
        "util_box":   fig.add_subplot(gs[2, 1]),
        "income_hist": fig.add_subplot(gs[2, 2]),
    }

    dark_bg = "#1A1D27"
    text_c = "#E0E0E0"
    grid_c = "#2E3250"

    def style_ax(ax, title=""):
        ax.set_facecolor(dark_bg)
        ax.tick_params(colors=text_c, labelsize=8)
        ax.spines[:].set_color(grid_c)
        ax.yaxis.label.set_color(text_c)
        ax.xaxis.label.set_color(text_c)
        ax.set_title(title, color=text_c, fontsize=10,
                     fontweight="bold", pad=8)
        ax.grid(True, color=grid_c, linewidth=0.5, alpha=0.6)

    # ── Panel 1: ROC Curves ──
    ax = axes["roc"]
    style_ax(ax, "ROC Curves")
    ax.plot([0, 1], [0, 1], "--", color="#555", linewidth=1)
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = roc_auc_score(y_test, y_prob)
        ax.plot(fpr, tpr, label=f"{name} ({auc:.3f})",
                color=PALETTE[name], linewidth=2)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.legend(fontsize=7, facecolor=dark_bg, labelcolor=text_c)

    # ── Panel 2: Precision-Recall Curves ──
    ax = axes["pr"]
    style_ax(ax, "Precision-Recall Curves")
    for name, model in models.items():
        y_prob = model.predict_proba(X_test)[:, 1]
        prec_c, rec_c, _ = precision_recall_curve(y_test, y_prob)
        ax.plot(rec_c, prec_c, label=name, color=PALETTE[name], linewidth=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(fontsize=7, facecolor=dark_bg, labelcolor=text_c)

    # ── Panel 3: Confusion Matrix (Best model = RF) ──
    ax = axes["cm"]
    best_model = models["Random Forest"]
    y_pred = best_model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Default", "Good Credit"],
                yticklabels=["Default", "Good Credit"],
                linewidths=0.5, linecolor=dark_bg)
    ax.set_facecolor(dark_bg)
    ax.tick_params(colors=text_c, labelsize=8)
    ax.set_title("Confusion Matrix (Random Forest)", color=text_c,
                 fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Predicted", color=text_c)
    ax.set_ylabel("Actual", color=text_c)

    # ── Panel 4: Metric Comparison Bar Chart ──
    ax = axes["metrics"]
    style_ax(ax, "Model Metrics Comparison")
    metrics_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x = np.arange(len(metrics_cols))
    width = 0.25
    for i, (_, row) in enumerate(summary_df.iterrows()):
        vals = [row[m] for m in metrics_cols]
        bars = ax.bar(x + i*width, vals, width, label=row["Model"],
                      color=list(PALETTE.values())[i], alpha=0.85)
    ax.set_xticks(x + width)
    ax.set_xticklabels(metrics_cols, color=text_c, fontsize=9)
    ax.set_ylim(0.5, 1.05)
    ax.legend(fontsize=8, facecolor=dark_bg, labelcolor=text_c)

    # ── Panel 5: Feature Importance (Random Forest) ──
    ax = axes["feat_imp"]
    style_ax(ax, "Feature Importance (RF)")
    importances = best_model.feature_importances_
    top_n = 10
    idx = np.argsort(importances)[-top_n:]
    ax.barh(np.array(feature_names)[idx], importances[idx],
            color="#4FD18A", alpha=0.85)
    ax.set_xlabel("Importance")

    # ── Panel 6: Score Distribution by Class ──
    ax = axes["dist"]
    style_ax(ax, "Predicted Probability Distribution")
    y_prob = best_model.predict_proba(X_test)[:, 1]
    ax.hist(y_prob[y_test == 0], bins=30, alpha=0.7, color="#F7584F",
            label="Default (0)", density=True)
    ax.hist(y_prob[y_test == 1], bins=30, alpha=0.7, color="#4FD18A",
            label="Good Credit (1)", density=True)
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, facecolor=dark_bg, labelcolor=text_c)

    # ── Panel 7: Credit Utilization by Class ──
    ax = axes["util_box"]
    style_ax(ax, "Credit Utilization by Class")
    good = df_raw.loc[df_raw["creditworthy"]
                      == 1, "credit_util_ratio"].dropna()
    bad = df_raw.loc[df_raw["creditworthy"] == 0, "credit_util_ratio"].dropna()
    bp = ax.boxplot([bad, good], labels=["Default", "Good Credit"],
                    patch_artist=True, medianprops=dict(color="white", linewidth=2))
    bp["boxes"][0].set_facecolor("#F7584F")
    bp["boxes"][1].set_facecolor("#4FD18A")
    ax.set_ylabel("Utilization Ratio")

    # ── Panel 8: Income Distribution by Class ──
    ax = axes["income_hist"]
    style_ax(ax, "Income Distribution by Class")
    df_raw["income"].fillna(df_raw["income"].median(), inplace=True)
    ax.hist(df_raw.loc[df_raw["creditworthy"] == 0, "income"], bins=30,
            alpha=0.7, color="#F7584F", label="Default", density=True)
    ax.hist(df_raw.loc[df_raw["creditworthy"] == 1, "income"], bins=30,
            alpha=0.7, color="#4FD18A", label="Good Credit", density=True)
    ax.set_xlabel("Annual Income (₹ / $)")
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, facecolor=dark_bg, labelcolor=text_c)

    plt.savefig("/home/claude/credit_scoring/dashboard.png",
                dpi=150, bbox_inches="tight", facecolor="#0F1117")
    print("\n  📊 Dashboard saved → credit_scoring/dashboard.png")
    plt.close()


# ─────────────────────────────────────────────
#  STEP 7 — PREDICT NEW APPLICANT
# ─────────────────────────────────────────────

def predict_applicant(model, imputer, scaler, feature_names: list):
    """
    Demo: predict creditworthiness for a sample new applicant.
    """
    sample = {
        "age":               35,
        "income":            62000,
        "employment_years":  8,
        "num_credit_lines":  5,
        "credit_util_ratio": 0.35,
        "num_late_payments": 1,
        "total_debt":        22000,
        "debt_to_income":    0.35,
        "loan_amount":       18000,
        "num_inquiries":     2,
        "loan_purpose_enc":  2,         # "education" encoded
        "has_mortgage":      0,
        "savings_balance":   8000,
        # engineered features
        "payment_discipline":  1 / (1+1),
        "income_per_year_emp": 62000 / (8+1),
        "loan_to_income":      18000 / (62000+1),
        "debt_burden_score":   0.35 * 0.35,
        "inquiry_density":     2 / (5+1),
        "savings_to_debt":     8000 / (22000+1),
        "risk_composite":      1*0.4 + 0.35*0.3 + 0.35*0.2 + 2*0.1,
    }

    applicant_df = pd.DataFrame([sample])[feature_names]
    X_imp = imputer.transform(applicant_df)
    X_sc = scaler.transform(X_imp)
    prob = model.predict_proba(X_sc)[0][1]
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
    print(f"  → Creditworthiness Probability : {prob:.4f} ({prob*100:.1f}%)")
    print(f"  → Decision                     : {label}")
    print("═"*60)


# ─────────────────────────────────────────────
#  MAIN PIPELINE
# ─────────────────────────────────────────────

if __name__ == "__main__":

    print("\n" + "═"*60)
    print("  CREDIT SCORING MODEL — AI/ML Task 1")
    print("  Logistic Regression | Decision Tree | Random Forest")
    print("═"*60)

    # 1. Data
    print("\n  [1/6] Generating synthetic dataset …")
    df_raw = generate_credit_dataset(n_samples=2000)
    print(
        f"       Shape: {df_raw.shape}  |  Default rate: {(1-df_raw['creditworthy'].mean()):.1%}")

    # 2. Feature engineering
    print("  [2/6] Feature engineering …")
    df_fe = feature_engineering(df_raw)

    # 3. Preprocess
    print("  [3/6] Preprocessing (impute → split → scale) …")
    X_train, X_test, X_train_raw, X_test_raw, y_train, y_test, \
        feature_names, imputer, scaler = preprocess(df_fe)
    print(f"       Train: {X_train.shape}  |  Test: {X_test.shape}")

    # 4. Train
    models = train_models(X_train, y_train)

    # 5. Evaluate
    summary = evaluate_models(models, X_test, y_test)
    print("\n  ── SUMMARY TABLE ──")
    print(summary.to_string(index=False))

    # 6. Visualise
    print("\n  [6/6] Generating dashboard …")
    plot_dashboard(models, X_test, y_test, feature_names, summary, df_raw)

    # 7. Predict a new applicant
    predict_applicant(models["Random Forest"], imputer, scaler, feature_names)

    print("\n  ✅ All done! Check credit_scoring/dashboard.png\n")
