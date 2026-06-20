"""
visualize.py
6-panel diagnostic dashboard for credit scoring models.
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_auc_score, roc_curve

PALETTE = {
    "Logistic Regression": "#4F8EF7",
    "Decision Tree":       "#F7A24F",
    "Random Forest":       "#4FD18A",
}
DARK_BG = "#1A1D27"
TEXT_C  = "#E0E0E0"
GRID_C  = "#2E3250"


def _style_ax(ax, title=""):
    ax.set_facecolor(DARK_BG)
    ax.tick_params(colors=TEXT_C, labelsize=8)
    ax.spines[:].set_color(GRID_C)
    ax.yaxis.label.set_color(TEXT_C)
    ax.xaxis.label.set_color(TEXT_C)
    ax.set_title(title, color=TEXT_C, fontsize=10, fontweight="bold", pad=8)
    ax.grid(True, color=GRID_C, linewidth=0.5, alpha=0.6)


def plot_dashboard(models, X_test, y_test, feature_names, summary_df, df_raw, out_path="dashboard.png"):
    fig = plt.figure(figsize=(20, 18), facecolor="#0F1117")
    fig.suptitle("CREDIT SCORING MODEL — DIAGNOSTIC DASHBOARD",
                 fontsize=18, fontweight="bold", color="white", y=0.98)

    gs   = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)
    axes = {
        "roc":         fig.add_subplot(gs[0, 0]),
        "pr":          fig.add_subplot(gs[0, 1]),
        "cm":          fig.add_subplot(gs[0, 2]),
        "metrics":     fig.add_subplot(gs[1, 0:2]),
        "feat_imp":    fig.add_subplot(gs[1, 2]),
        "dist":        fig.add_subplot(gs[2, 0]),
        "util_box":    fig.add_subplot(gs[2, 1]),
        "income_hist": fig.add_subplot(gs[2, 2]),
    }

    best = models["Random Forest"]

    # ROC
    ax = axes["roc"]; _style_ax(ax, "ROC Curves")
    ax.plot([0, 1], [0, 1], "--", color="#555", linewidth=1)
    for name, model in models.items():
        fpr, tpr, _ = roc_curve(y_test, model.predict_proba(X_test)[:, 1])
        ax.plot(fpr, tpr, label=f"{name} ({roc_auc_score(y_test, model.predict_proba(X_test)[:,1]):.3f})",
                color=PALETTE[name], linewidth=2)
    ax.set_xlabel("FPR"); ax.set_ylabel("TPR")
    ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor=TEXT_C)

    # PR
    ax = axes["pr"]; _style_ax(ax, "Precision-Recall Curves")
    for name, model in models.items():
        p, r, _ = precision_recall_curve(y_test, model.predict_proba(X_test)[:, 1])
        ax.plot(r, p, label=name, color=PALETTE[name], linewidth=2)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.legend(fontsize=7, facecolor=DARK_BG, labelcolor=TEXT_C)

    # Confusion matrix
    ax = axes["cm"]
    cm = confusion_matrix(y_test, best.predict(X_test))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Default", "Good Credit"],
                yticklabels=["Default", "Good Credit"],
                linewidths=0.5, linecolor=DARK_BG)
    ax.set_facecolor(DARK_BG); ax.tick_params(colors=TEXT_C, labelsize=8)
    ax.set_title("Confusion Matrix (RF)", color=TEXT_C, fontsize=10, fontweight="bold", pad=8)
    ax.set_xlabel("Predicted", color=TEXT_C); ax.set_ylabel("Actual", color=TEXT_C)

    # Metrics bar
    ax = axes["metrics"]; _style_ax(ax, "Model Metrics Comparison")
    metrics_cols = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]
    x = np.arange(len(metrics_cols)); w = 0.25
    for i, (_, row) in enumerate(summary_df.iterrows()):
        ax.bar(x + i*w, [row[m] for m in metrics_cols], w,
               label=row["Model"], color=list(PALETTE.values())[i], alpha=0.85)
    ax.set_xticks(x + w); ax.set_xticklabels(metrics_cols, color=TEXT_C, fontsize=9)
    ax.set_ylim(0.5, 1.05); ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_C)

    # Feature importance
    ax = axes["feat_imp"]; _style_ax(ax, "Feature Importance (RF)")
    imp = best.feature_importances_; idx = np.argsort(imp)[-10:]
    ax.barh(np.array(feature_names)[idx], imp[idx], color="#4FD18A", alpha=0.85)
    ax.set_xlabel("Importance")

    # Score distribution
    ax = axes["dist"]; _style_ax(ax, "Predicted Probability Distribution")
    y_prob = best.predict_proba(X_test)[:, 1]
    ax.hist(y_prob[y_test == 0], bins=30, alpha=0.7, color="#F7584F", label="Default", density=True)
    ax.hist(y_prob[y_test == 1], bins=30, alpha=0.7, color="#4FD18A", label="Good Credit", density=True)
    ax.set_xlabel("Predicted Probability"); ax.set_ylabel("Density")
    ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_C)

    # Utilization boxplot
    ax = axes["util_box"]; _style_ax(ax, "Credit Utilization by Class")
    bp = ax.boxplot(
        [df_raw.loc[df_raw["creditworthy"]==0, "credit_util_ratio"].dropna(),
         df_raw.loc[df_raw["creditworthy"]==1, "credit_util_ratio"].dropna()],
        labels=["Default", "Good Credit"], patch_artist=True,
        medianprops=dict(color="white", linewidth=2))
    bp["boxes"][0].set_facecolor("#F7584F"); bp["boxes"][1].set_facecolor("#4FD18A")
    ax.set_ylabel("Utilization Ratio")

    # Income histogram
    ax = axes["income_hist"]; _style_ax(ax, "Income Distribution by Class")
    df_raw["income"].fillna(df_raw["income"].median(), inplace=True)
    ax.hist(df_raw.loc[df_raw["creditworthy"]==0, "income"], bins=30, alpha=0.7, color="#F7584F", label="Default", density=True)
    ax.hist(df_raw.loc[df_raw["creditworthy"]==1, "income"], bins=30, alpha=0.7, color="#4FD18A", label="Good Credit", density=True)
    ax.set_xlabel("Annual Income"); ax.set_ylabel("Density")
    ax.legend(fontsize=8, facecolor=DARK_BG, labelcolor=TEXT_C)

    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0F1117")
    print(f"\n  📊 Dashboard saved → {out_path}")
    plt.close()
