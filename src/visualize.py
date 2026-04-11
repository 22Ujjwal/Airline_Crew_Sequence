"""
Pair-centric visualizations with statistical significance.
All plots treat (airport_A, airport_B) as the unit of analysis.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix, ConfusionMatrixDisplay

sys.path.insert(0, os.path.dirname(__file__))
from significance import compute_significance, top_significant_pairs, wilson_ci, MIN_N

PROC  = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
PLOTS = os.path.join(PROC, "plots")
os.makedirs(PLOTS, exist_ok=True)

FEATURE_COLS = [
    "A_weather_delay_rate","A_weather_cancel_rate","A_avg_weather_delay_min",
    "A_p75_weather_delay_min","A_p95_weather_delay_min","A_nas_delay_rate",
    "A_overall_weather_delay_rate","A_overall_avg_weather_delay_min",
    "B_weather_delay_rate","B_weather_cancel_rate","B_avg_weather_delay_min",
    "B_p75_weather_delay_min","B_p95_weather_delay_min","B_nas_delay_rate",
    "B_overall_weather_delay_rate","B_overall_avg_weather_delay_min",
    "pair_combined_weather_rate","pair_max_weather_rate","pair_min_weather_rate",
    "pair_weather_rate_sum","pair_avg_weather_delay_min","both_high_risk",
    "Month","is_spring_summer","median_turnaround_min",
]
MONTH_NAMES = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

def save(fig, name):
    path = os.path.join(PLOTS, f"{name}.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── 01: Pair risk scores with Wilson CI error bars ───────────────────────────
def plot_pair_risk_ci(sig_df):
    top = top_significant_pairs(sig_df, n=25)
    top["pair"] = top["airport_A"] + " → DFW → " + top["airport_B"]
    top = top.iloc[::-1]  # highest at top

    fig, ax = plt.subplots(figsize=(12, 9))
    y = np.arange(len(top))

    ax.barh(y, top["avg_bad_rate"], color="coral", alpha=0.75, label="Observed bad rate")
    ax.barh(y, top["avg_risk_score"], color="steelblue", alpha=0.55, label="Model risk score")

    # Error bars on observed rate (Wilson CI)
    xerr_lo = top["avg_bad_rate"] - top["avg_ci_lower"]
    xerr_hi = top["avg_ci_upper"] - top["avg_bad_rate"]
    ax.errorbar(top["avg_bad_rate"], y,
                xerr=[xerr_lo, xerr_hi],
                fmt="none", color="darkred", capsize=3, linewidth=1.2, label="95% Wilson CI")

    ax.set_yticks(y)
    ax.set_yticklabels(top["pair"], fontsize=9)
    ax.set_xlabel("Risk score / Observed bad rate", fontsize=11)
    ax.set_xlim(0, 1)
    ax.set_title(
        f"Top 25 Statistically Significant High-Risk Pairs\n"
        f"(n ≥ {MIN_N}, FDR-corrected α=0.05 | error bars = 95% Wilson CI)",
        fontsize=13
    )
    ax.axvline(sig_df["baseline_rate"].iloc[0], color="gray", linestyle="--",
               linewidth=1, label=f"Baseline ({sig_df['baseline_rate'].iloc[0]:.1%})")
    ax.legend(loc="lower right", fontsize=9)
    # Annotate risk ratio
    for i, row in enumerate(top.itertuples()):
        ax.text(max(row.avg_risk_score, row.avg_bad_rate) + 0.01, i,
                f"×{row.risk_ratio:.1f}", va="center", fontsize=7, color="dimgray")
    save(fig, "01_pair_risk_ci")


# ── 02: Pair risk heatmap with significance overlay ──────────────────────────
def plot_pair_heatmap_sig(sig_df):
    # Aggregate across months: mean risk score, count significant months
    pair_agg = (
        sig_df.groupby(["airport_A","airport_B"])
        .agg(
            avg_risk     = ("avg_risk_score",    "mean"),
            sig_months   = ("significant",        "sum"),
            total_n      = ("n_sequences",        "sum"),
            bad_rate     = ("observed_bad_rate",  "mean"),
        )
        .reset_index()
    )
    # Filter: at least 1 significant month and enough data
    pair_agg = pair_agg[(pair_agg["total_n"] >= MIN_N)]

    top_a = pair_agg.groupby("airport_A")["avg_risk"].mean()\
                    .sort_values(ascending=False).head(15).index.tolist()
    top_b = pair_agg.groupby("airport_B")["avg_risk"].mean()\
                    .sort_values(ascending=False).head(15).index.tolist()

    sub = pair_agg[pair_agg["airport_A"].isin(top_a) & pair_agg["airport_B"].isin(top_b)]
    pivot_risk = sub.pivot(index="airport_A", columns="airport_B", values="avg_risk")\
                    .reindex(index=top_a, columns=top_b)
    pivot_sig  = sub.pivot(index="airport_A", columns="airport_B", values="sig_months")\
                    .reindex(index=top_a, columns=top_b).fillna(0)

    fig, ax = plt.subplots(figsize=(14, 10))
    im = ax.imshow(pivot_risk.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(top_b)))
    ax.set_xticklabels(top_b, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(top_a)))
    ax.set_yticklabels(top_a, fontsize=9)

    for i in range(len(top_a)):
        for j in range(len(top_b)):
            val = pivot_risk.values[i, j]
            sig = pivot_sig.values[i, j]
            if np.isnan(val):
                continue
            txt = f"{val:.2f}"
            color = "white" if val > 0.6 else "black"
            t = ax.text(j, i, txt, ha="center", va="center", fontsize=7.5, color=color)
            # Star if significant in ≥1 month
            if sig >= 1:
                ax.text(j + 0.35, i - 0.35, "★", ha="center", va="center",
                        fontsize=7, color="gold",
                        path_effects=[pe.withStroke(linewidth=1, foreground="black")])

    plt.colorbar(im, ax=ax, label="Avg Risk Score", shrink=0.8)
    ax.set_title(
        "Pair Risk Heatmap — Avg Risk Score\n"
        "★ = statistically significant in ≥1 month (FDR-adjusted)",
        fontsize=13
    )
    ax.set_xlabel("Airport B (outbound from DFW)", fontsize=11)
    ax.set_ylabel("Airport A (inbound to DFW)", fontsize=11)
    save(fig, "02_pair_heatmap_sig")


# ── 03: Volcano plot — effect size vs significance ──────────────────────────
def plot_volcano(sig_df):
    df = sig_df[(sig_df["n_sequences"] >= MIN_N) & sig_df["p_value"].notna()].copy()
    df["neg_log_p"] = -np.log10(df["p_value"].clip(1e-300))

    fig, ax = plt.subplots(figsize=(10, 7))

    # Non-significant
    mask_ns  = ~df["significant"]
    ax.scatter(df.loc[mask_ns, "effect_size"], df.loc[mask_ns, "neg_log_p"],
               c="lightgray", s=6, alpha=0.4, label="Not significant")

    # Significant high-risk
    mask_hr  = df["significant"] & (df["risk_ratio"] > 1)
    sc = ax.scatter(df.loc[mask_hr, "effect_size"], df.loc[mask_hr, "neg_log_p"],
                    c=df.loc[mask_hr, "avg_risk_score"], cmap="YlOrRd",
                    s=20, alpha=0.8, vmin=0, vmax=1, label="Significant high-risk")
    plt.colorbar(sc, ax=ax, label="Model risk score", shrink=0.7)

    # Threshold lines
    sig_threshold = -np.log10(0.05)
    ax.axhline(sig_threshold, color="gray", linestyle="--", linewidth=0.8)
    ax.axvline(0, color="gray", linestyle="--", linewidth=0.8)

    # Label top 10
    top10 = df[mask_hr].nlargest(10, "neg_log_p")
    for _, row in top10.iterrows():
        ax.annotate(f"{row['airport_A']}→{row['airport_B']}",
                    (row["effect_size"], row["neg_log_p"]),
                    textcoords="offset points", xytext=(4, 2),
                    fontsize=6.5, color="darkred")

    ax.set_xlabel("Effect size  (risk ratio − 1)", fontsize=11)
    ax.set_ylabel("−log₁₀(p-value)", fontsize=11)
    ax.set_title("Volcano Plot: Pair Risk vs Statistical Significance\n"
                 "(right = higher risk than baseline, up = more significant)", fontsize=13)
    ax.legend(fontsize=9)
    save(fig, "03_volcano_plot")


# ── 04: Monthly risk profiles — top 6 pairs ─────────────────────────────────
def plot_monthly_profiles(sig_df):
    top6 = top_significant_pairs(sig_df, n=6)

    fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharey=False)
    axes = axes.flatten()
    baseline = sig_df["baseline_rate"].iloc[0]

    for ax, (_, row) in zip(axes, top6.iterrows()):
        a, b = row["airport_A"], row["airport_B"]
        monthly = sig_df[
            (sig_df["airport_A"] == a) & (sig_df["airport_B"] == b)
        ].sort_values("Month")

        months  = monthly["Month"].values
        scores  = monthly["avg_risk_score"].values
        bad_rt  = monthly["observed_bad_rate"].values
        ci_lo   = monthly["ci_lower"].values
        ci_hi   = monthly["ci_upper"].values
        sig     = monthly["significant"].values

        x = np.arange(len(months))
        ax.bar(x, bad_rt, color=["coral" if s else "lightcoral" for s in sig],
               alpha=0.8, label="Observed bad rate")
        ax.plot(x, scores, "o--", color="steelblue", linewidth=1.5,
                markersize=4, label="Model score")
        ax.fill_between(x, ci_lo, ci_hi, alpha=0.2, color="darkred", label="95% CI")
        ax.axhline(baseline, color="gray", linestyle=":", linewidth=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels([MONTH_NAMES[m-1] for m in months], fontsize=7, rotation=45)
        ax.set_ylim(0, 1)
        ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
        n_sig = sig.sum()
        ax.set_title(f"{a} → DFW → {b}\n({n_sig}/{len(months)} months significant)",
                     fontsize=10, fontweight="bold")

    handles = [
        mpatches.Patch(color="coral",     label="Observed bad rate (★ sig)"),
        mpatches.Patch(color="lightcoral",label="Observed bad rate (ns)"),
        plt.Line2D([0],[0], color="steelblue", marker="o", label="Model score"),
        mpatches.Patch(color="darkred", alpha=0.2, label="95% Wilson CI"),
        plt.Line2D([0],[0], color="gray", linestyle=":", label="Baseline"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8, bbox_to_anchor=(0.5,-0.02))
    fig.suptitle("Monthly Risk Profiles — Top 6 High-Risk Pairs\n"
                 "(coral = statistically significant months)", fontsize=13, y=1.01)
    plt.tight_layout()
    save(fig, "04_monthly_profiles")


# ── 05: Seasonality — pair-level bad rate by month ──────────────────────────
def plot_seasonality_pair(sig_df):
    # For each month: distribution of pair bad rates (violin)
    months_present = sorted(sig_df["Month"].unique())
    data_by_month = [
        sig_df[(sig_df["Month"] == m) & (sig_df["n_sequences"] >= MIN_N)]["observed_bad_rate"].values
        for m in months_present
    ]
    baseline = sig_df["baseline_rate"].iloc[0]

    fig, ax = plt.subplots(figsize=(13, 5))
    parts = ax.violinplot(data_by_month, positions=months_present,
                          showmedians=True, showextrema=False)
    for pc in parts["bodies"]:
        pc.set_facecolor("steelblue")
        pc.set_alpha(0.6)
    parts["cmedians"].set_color("darkblue")

    ax.axhline(baseline, color="coral", linestyle="--", linewidth=1.2,
               label=f"Overall baseline ({baseline:.1%})")
    ax.set_xticks(months_present)
    ax.set_xticklabels([MONTH_NAMES[m-1] for m in months_present])
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    ax.set_ylabel("Observed bad rate across pairs", fontsize=11)
    ax.set_title("Distribution of Pair Bad Rates by Month\n"
                 "(each violin = all (A,B) pairs in that month; median line shown)", fontsize=13)
    ax.legend()
    save(fig, "05_seasonality_pair")


# ── 06: Risk ratio distribution — how bad are the bad pairs? ────────────────
def plot_risk_ratio_dist(sig_df):
    df = sig_df[sig_df["n_sequences"] >= MIN_N].copy()
    baseline = df["baseline_rate"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: histogram of risk ratios
    axes[0].hist(df["risk_ratio"], bins=60, color="steelblue", edgecolor="white", linewidth=0.3)
    axes[0].axvline(1.0, color="coral", linestyle="--", linewidth=1.5, label="Baseline (RR=1)")
    axes[0].axvline(df[df["significant"]]["risk_ratio"].median(), color="darkred",
                    linestyle="-", linewidth=1.2,
                    label=f"Median significant RR ({df[df['significant']]['risk_ratio'].median():.2f})")
    axes[0].set_xlabel("Risk Ratio (pair bad rate / baseline)", fontsize=11)
    axes[0].set_ylabel("Count of pair×month combinations", fontsize=11)
    axes[0].set_title("Distribution of Risk Ratios\n(pairs with n ≥ 30)", fontsize=12)
    axes[0].legend(fontsize=9)

    # Right: model score vs risk ratio scatter, colored by significance
    sig_mask = df["significant"]
    axes[1].scatter(df.loc[~sig_mask, "avg_risk_score"], df.loc[~sig_mask, "risk_ratio"],
                    c="lightgray", s=5, alpha=0.3, label="Not significant")
    sc = axes[1].scatter(df.loc[sig_mask, "avg_risk_score"], df.loc[sig_mask, "risk_ratio"],
                         c=df.loc[sig_mask, "observed_bad_rate"], cmap="YlOrRd",
                         s=15, alpha=0.7, vmin=0, vmax=1, label="Significant")
    plt.colorbar(sc, ax=axes[1], label="Observed bad rate", shrink=0.8)
    axes[1].axhline(1.0, color="gray", linestyle="--", linewidth=0.8)
    axes[1].set_xlabel("Model risk score", fontsize=11)
    axes[1].set_ylabel("Risk ratio vs baseline", fontsize=11)
    axes[1].set_title("Model Score vs Risk Ratio\n(colored by observed bad rate)", fontsize=12)
    axes[1].legend(fontsize=9)

    fig.suptitle("Pair Risk Ratio Analysis", fontsize=13)
    save(fig, "06_risk_ratio_dist")


# ── 07: Model evaluation (ROC / PR / confusion matrix) ──────────────────────
def plot_model_eval(df, model, feature_cols):
    val = df[df["Year"] == df["Year"].max()].copy()
    present = [c for c in feature_cols if c in val.columns]
    X = val[present].astype(float)
    mask = X.notna().all(axis=1)
    X, y = X[mask], val["target"][mask]

    proba = model.predict_proba(X)[:, 1]
    pred  = model.predict(X)

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    RocCurveDisplay.from_predictions(y, proba, ax=axes[0], name="XGBoost",
                                     color="coral", plot_chance_level=True)
    axes[0].set_title(f"ROC Curve — AUC={roc_auc_score(y,proba):.3f}", fontsize=12)
    PrecisionRecallDisplay.from_predictions(y, proba, ax=axes[1], name="XGBoost", color="steelblue")
    axes[1].set_title(f"Precision–Recall — AP={average_precision_score(y,proba):.3f}", fontsize=12)
    ConfusionMatrixDisplay(confusion_matrix(y,pred), display_labels=["low","high"])\
        .plot(ax=axes[2], colorbar=False, cmap="Blues")
    axes[2].set_title("Confusion Matrix (2024 holdout)", fontsize=12)
    fig.suptitle("XGBoost Model Evaluation — Holdout Year", fontsize=14, y=1.01)
    save(fig, "07_model_evaluation")


# ── 08: Feature importance (pair-centric framing) ───────────────────────────
def plot_feature_importance(model, feature_cols):
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"feature": feature_cols, "importance": importances})\
                .sort_values("importance", ascending=True).tail(20)

    colors = [
        "coral"     if any(k in f for k in ("pair","both")) else
        "steelblue" if f.startswith("A_") else
        "seagreen"  if f.startswith("B_") else
        "orchid"
        for f in feat_df["feature"]
    ]
    fig, ax = plt.subplots(figsize=(11, 7))
    ax.barh(feat_df["feature"], feat_df["importance"], color=colors)
    ax.set_xlabel("Feature importance (gain)", fontsize=12)
    ax.set_title("Top 20 Feature Importances\n(Pair-level features are the strongest predictors)", fontsize=13)
    legend_handles = [
        mpatches.Patch(color="coral",     label="Pair-level (A×B combined)"),
        mpatches.Patch(color="steelblue", label="Airport A (inbound leg)"),
        mpatches.Patch(color="seagreen",  label="Airport B (outbound leg)"),
        mpatches.Patch(color="orchid",    label="Temporal / turnaround"),
    ]
    ax.legend(handles=legend_handles, loc="lower right", fontsize=9)
    save(fig, "08_feature_importance")


# ── 09: Calibration ─────────────────────────────────────────────────────────
def plot_calibration(df, model, feature_cols):
    present = [c for c in feature_cols if c in df.columns]
    X = df[present].astype(float)
    mask = X.notna().all(axis=1)
    X, y = X[mask], df["target"][mask]
    proba = model.predict_proba(X)[:, 1]

    cal = pd.DataFrame({"score": proba, "actual": y})
    cal["decile"] = pd.qcut(cal["score"], q=10, labels=False, duplicates="drop")
    calib = cal.groupby("decile").agg(
        avg_pred=("score","mean"), actual_rate=("actual","mean"), n=("actual","count")
    ).reset_index()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.plot([0,1],[0,1],"--", color="gray", label="Perfect calibration")
    ax.scatter(calib["avg_pred"], calib["actual_rate"],
               s=calib["n"]/calib["n"].max()*400,
               c=calib["avg_pred"], cmap="YlOrRd", zorder=5,
               edgecolors="k", linewidths=0.5)
    for _, r in calib.iterrows():
        ax.annotate(f"n={r['n']:,.0f}", (r["avg_pred"],r["actual_rate"]),
                    textcoords="offset points", xytext=(5,3), fontsize=7)
    ax.set_xlabel("Avg predicted score (by decile)", fontsize=12)
    ax.set_ylabel("Actual high-risk rate", fontsize=12)
    ax.set_title("Model Calibration\n(dots near diagonal = well-calibrated)", fontsize=13)
    ax.legend()
    save(fig, "09_calibration")


# ── 10: Score distribution ───────────────────────────────────────────────────
def plot_score_distribution(df, model, feature_cols):
    val = df[df["Year"] == df["Year"].max()].copy()
    present = [c for c in feature_cols if c in val.columns]
    X = val[present].astype(float)
    mask = X.notna().all(axis=1)
    X, y = X[mask], val["target"][mask]
    proba = model.predict_proba(X)[:, 1]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(proba[y==0], bins=60, alpha=0.65, label="Low-risk (actual)",  color="steelblue", density=True)
    ax.hist(proba[y==1], bins=60, alpha=0.65, label="High-risk (actual)", color="coral",     density=True)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=1, label="Decision boundary (0.5)")
    ax.set_xlabel("Predicted risk score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Risk Score Distribution by True Label\n"
                 "(better separation = stronger model)", fontsize=13)
    ax.legend()
    save(fig, "10_score_distribution")


# ── 11: Turnaround window risk interaction ──────────────────────────────────
def plot_turnaround_risk(sig_df):
    df = sig_df[sig_df["n_sequences"] >= MIN_N].copy()
    if "median_turnaround_min" not in df.columns:
        return
    bins   = [30, 60, 90, 120, 180, 240]
    labels = ["30–60","60–90","90–120","120–180","180–240"]
    df["t_bin"] = pd.cut(df["median_turnaround_min"], bins=bins, labels=labels)
    risk_bins  = df.groupby(["t_bin","Month"], observed=True)["observed_bad_rate"].mean().unstack()

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: bad rate by turnaround bin
    t_agg = df.groupby("t_bin", observed=True).agg(
        bad_rate=("observed_bad_rate","mean"),
        n=("n_sequences","sum"),
    ).reset_index()
    axes[0].bar(t_agg["t_bin"], t_agg["bad_rate"], color="steelblue")
    axes[0].yaxis.set_major_formatter(mtick.PercentFormatter(1.0))
    axes[0].set_xlabel("Turnaround window (minutes)", fontsize=11)
    axes[0].set_ylabel("Avg observed bad rate", fontsize=11)
    axes[0].set_title("Risk vs Turnaround Window\n(tight turnarounds = less margin for delay cascade)",
                      fontsize=12)

    # Right: heatmap — turnaround bin × month
    if not risk_bins.empty:
        im = axes[1].imshow(risk_bins.values, cmap="YlOrRd", aspect="auto", vmin=0, vmax=0.6)
        axes[1].set_yticks(range(len(risk_bins.index)))
        axes[1].set_yticklabels(risk_bins.index, fontsize=9)
        axes[1].set_xticks(range(len(risk_bins.columns)))
        axes[1].set_xticklabels([MONTH_NAMES[int(m)-1] for m in risk_bins.columns], fontsize=8)
        plt.colorbar(im, ax=axes[1], label="Avg bad rate", shrink=0.8)
        axes[1].set_title("Bad Rate: Turnaround Window × Month", fontsize=12)

    fig.suptitle("Turnaround Window × Risk Interaction", fontsize=13)
    save(fig, "11_turnaround_risk")


# ── 12: Pair bad rate vs combined weather rate (pair space) ─────────────────
def plot_pair_weather_risk(sig_df):
    df = sig_df[sig_df["n_sequences"] >= MIN_N].copy()

    fig, ax = plt.subplots(figsize=(9, 7))
    ns_mask = ~df["significant"]
    s_mask  = df["significant"]

    ax.scatter(df.loc[ns_mask,"pair_combined_weather_rate"],
               df.loc[ns_mask,"observed_bad_rate"],
               c="lightgray", s=6, alpha=0.25, label="Not significant")
    sc = ax.scatter(df.loc[s_mask,"pair_combined_weather_rate"],
                    df.loc[s_mask,"observed_bad_rate"],
                    c=df.loc[s_mask,"avg_risk_score"], cmap="YlOrRd",
                    s=18, alpha=0.75, vmin=0, vmax=1, label="Significant")
    plt.colorbar(sc, ax=ax, label="Model risk score", shrink=0.8)

    # Trend line on significant pairs only
    if s_mask.sum() > 2:
        z = np.polyfit(df.loc[s_mask,"pair_combined_weather_rate"],
                       df.loc[s_mask,"observed_bad_rate"], 1)
        xs = np.linspace(df["pair_combined_weather_rate"].min(),
                         df["pair_combined_weather_rate"].max(), 100)
        ax.plot(xs, np.poly1d(z)(xs), "k--", linewidth=1.2, label="Trend (sig. pairs)")

    ax.set_xlabel("Pair combined weather rate (A_rate × B_rate)", fontsize=11)
    ax.set_ylabel("Observed bad rate", fontsize=11)
    ax.set_title("Pair Combined Weather Risk vs Observed Bad Rate\n"
                 "(both airports high-risk = top-right = worst sequences)", fontsize=13)
    ax.legend(fontsize=9)
    save(fig, "12_pair_weather_risk")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading data...")
    df          = pd.read_parquet(os.path.join(PROC, "sequence_features.parquet"))
    pair_scores = pd.read_parquet(os.path.join(PROC, "pair_risk_scores.parquet"))
    model       = xgb.XGBClassifier()
    model.load_model(os.path.join(PROC, "xgb_model.json"))

    season_cols  = [c for c in df.columns if c.startswith("season_")]
    feature_cols = FEATURE_COLS + season_cols
    feature_cols = [c for c in feature_cols if c in df.columns]

    print("\nComputing statistical significance...")
    sig_df = compute_significance(pair_scores)

    # Enrich sig_df with pair-level feature columns from sequence_features
    pair_feat_cols = ["airport_A", "airport_B", "Month",
                      "pair_combined_weather_rate", "pair_max_weather_rate",
                      "pair_weather_rate_sum", "median_turnaround_min"]
    pair_feat_cols = [c for c in pair_feat_cols if c in df.columns]
    pair_feats = df[pair_feat_cols].drop_duplicates(subset=["airport_A","airport_B","Month"])
    sig_df = sig_df.merge(pair_feats, on=["airport_A","airport_B","Month"], how="left")

    sig_df.to_parquet(os.path.join(PROC, "pair_scores_with_significance.parquet"), index=False)

    print("\nGenerating plots...")
    plot_pair_risk_ci(sig_df)
    plot_pair_heatmap_sig(sig_df)
    plot_volcano(sig_df)
    plot_monthly_profiles(sig_df)
    plot_seasonality_pair(sig_df)
    plot_risk_ratio_dist(sig_df)
    plot_model_eval(df, model, feature_cols)
    plot_feature_importance(model, feature_cols)
    plot_calibration(df, model, feature_cols)
    plot_score_distribution(df, model, feature_cols)
    plot_turnaround_risk(sig_df)
    plot_pair_weather_risk(sig_df)

    print(f"\nAll 12 plots saved to {PLOTS}/")


if __name__ == "__main__":
    main()
