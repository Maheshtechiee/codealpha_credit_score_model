"""
main.py
Entry point — runs the full credit scoring pipeline.
"""

import warnings
warnings.filterwarnings("ignore")

from src.data_generator import generate_credit_dataset
from src.features       import feature_engineering
from src.preprocessing  import preprocess
from src.train          import train_models
from src.evaluate       import evaluate_models
from src.visualize      import plot_dashboard
from src.predict        import predict_applicant


if __name__ == "__main__":
    print("\n" + "═"*60)
    print("  CREDIT SCORING MODEL — AI/ML Portfolio Project")
    print("  Logistic Regression | Decision Tree | Random Forest")
    print("═"*60)

    print("\n  [1/6] Generating synthetic dataset …")
    df_raw = generate_credit_dataset(n_samples=2000)
    print(f"       Shape: {df_raw.shape}  |  Default rate: {(1-df_raw['creditworthy'].mean()):.1%}")

    print("  [2/6] Feature engineering …")
    df_fe = feature_engineering(df_raw)

    print("  [3/6] Preprocessing (impute → split → scale) …")
    X_train, X_test, _, _, y_train, y_test, feature_names, imputer, scaler = preprocess(df_fe)
    print(f"       Train: {X_train.shape}  |  Test: {X_test.shape}")

    models  = train_models(X_train, y_train)
    summary = evaluate_models(models, X_test, y_test)

    print("\n  ── SUMMARY TABLE ──")
    print(summary.to_string(index=False))

    print("\n  [6/6] Generating dashboard …")
    plot_dashboard(models, X_test, y_test, feature_names, summary, df_raw, out_path="dashboard.png")

    predict_applicant(models["Random Forest"], imputer, scaler, feature_names)

    print("\n  ✅ All done!\n")
