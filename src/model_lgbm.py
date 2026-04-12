"""
LightGBM duty-aware crew sequence risk model (alt-model-duty-aware branch).

Differences from base XGBoost model:
  - LightGBM with focal loss → better handling of rare severe-weather events
  - Adds duty/fatigue/cascade features from feature_engineering_duty.py
  - TimeSeriesSplit cross-validation (not just single year holdout)
  - Outputs separate model artifact so base model is untouched

Focal loss motivation (from PDF): "What issues might arise from the sparsity of
severe weather events?" — standard cross-entropy underweights rare catastrophic
sequences. Focal loss down-weights easy negatives, focuses training on hard/rare
positives.
"""

import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report
from sklearn.model_selection import TimeSeriesSplit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))

PROC_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
PLOTS    = os.path.join(PROC_DIR, "plots")
os.makedirs(PLOTS, exist_ok=True)

RISK_THRESHOLD = 0.25  # matches model.py — absolute bad-rate threshold

# Base features (same as XGBoost)
BASE_FEATURES = [
    "A_weather_delay_rate", "A_weather_cancel_rate", "A_avg_weather_delay_min",
    "A_p75_weather_delay_min", "A_p95_weather_delay_min", "A_nas_delay_rate",
    "A_overall_weather_delay_rate", "A_overall_avg_weather_delay_min",
    "B_weather_delay_rate", "B_weather_cancel_rate", "B_avg_weather_delay_min",
    "B_p75_weather_delay_min", "B_p95_weather_delay_min", "B_nas_delay_rate",
    "B_overall_weather_delay_rate", "B_overall_avg_weather_delay_min",
    "pair_combined_weather_rate", "pair_max_weather_rate", "pair_min_weather_rate",
    "pair_weather_rate_sum", "pair_avg_weather_delay_min", "both_high_risk",
    "Month", "is_spring_summer", "median_turnaround_min",
    # GSOM wind + precip features
    "A_avg_wind_speed", "A_precip_days", "A_extreme_precip",
    "A_total_precip", "A_max_wind_gust",
    "B_avg_wind_speed", "B_precip_days", "B_extreme_precip",
    "B_total_precip", "B_max_wind_gust",
    "pair_max_avg_wind_speed", "pair_max_precip_days",
    "pair_max_extreme_precip", "pair_max_total_precip", "pair_max_max_wind_gust",
    # DFW hub weather
    "DFW_weather_delay_rate", "DFW_weather_cancel_rate",
    "DFW_avg_weather_delay_min", "DFW_p95_weather_delay_min",
]

# New duty/fatigue/cascade features
DUTY_FEATURES = [
    "A_dep_hour_median", "A_late_dep_rate", "A_early_dep_rate",
    "A_avg_block_min", "A_late_aircraft_delay_rate", "A_avg_late_aircraft_min",
    "B_dep_hour_median", "B_late_dep_rate", "B_early_dep_rate",
    "B_avg_block_min", "B_late_aircraft_delay_rate", "B_avg_late_aircraft_min",
    "tight_connection_rate", "very_tight_rate",
    "cascade_risk", "total_duty_block_min", "duty_overrun_risk", "late_dep_sequence",
]

# Tail-chain features: crew duty context from aircraft rotation tracking
TAIL_CHAIN_FEATURES = [
    "tc_legs_before_mean",    # avg legs flown before A→DFW leg
    "tc_block_before_mean",   # avg accumulated block time (min) before sequence
    "tc_duty_start_hour",     # median duty start hour (proxy for report time)
    "tc_total_duty_mean",     # avg full duty window (min): first dep → B arr
    "tc_total_duty_p75",      # p75 duty window (captures worst-case days)
    "tc_fdp_util_mean",       # avg FAA Part 117 FDP utilization (0–1.5)
    "tc_fdp_util_p75",        # p75 FDP utilization
    "tc_fdp_overrun_rate",    # fraction of sequences where FDP > limit
    "tc_wocl_rate",           # fraction crossing WOCL (0200–0559) circadian low
    "tc_legs_after_mean",     # avg legs remaining after B leg (scheduling pressure)
    "tc_legs_in_day_mean",    # avg total legs in duty day
    # Downstream cascade: A→DFW→B→DFW three-hop chain
    "tc_downstream_rate",     # fraction of sequences with matched B→DFW C leg
    "tc_cascade_late_rate",   # fraction of C legs with late aircraft delay > 15 min
    "tc_cascade_late_min",    # avg late aircraft delay on C leg (min)
    "tc_cascade_amplif_mean", # avg delay amplification ratio (C_delay / B_arr_delay)
    # Airport-level cascade propagation (A and B prefixed, joined below)
    "A_ap_cascade_rate",      # fraction of A's outbound flights with late aircraft delay
    "A_ap_cascade_given_late",# P(A outbound late | inbound to A was late)
    "B_ap_cascade_rate",      # fraction of B's outbound flights with late aircraft delay
    "B_ap_cascade_given_late",# P(B outbound late | inbound to B was late)
    "pair_cascade_product",   # A × B cascade rates (joint amplification)
    "pair_max_cascade_rate",  # max(A, B) cascade rate
    # Multi-hop DFW cascade: A→DFW→B→DFW→C→DFW→D (hub-and-spoke chain, existing data)
    "mhc_n_hops_mean",            # avg subsequent DFW hops after DFW→B
    "mhc_n_hops_max",             # max hops observed
    "mhc_total_late_min_mean",    # avg total downstream late aircraft delay (min)
    "mhc_total_late_min_p75",     # p75 total downstream delay
    "mhc_cascade_hop_rate",       # fraction with ≥1 downstream cascaded hop
    "mhc_cascade_depth_mean",     # avg number of hops with late aircraft delay
    "mhc_unique_airports_mean",   # avg unique airports hit downstream
    "mhc_recovery_rate",          # fraction where cascade fully clears
    # Full cascade chain features (available after download_bts_full.py + cascade_chain_features.py)
    "cc_chain_depth_mean",        # avg downstream legs after DFW→B
    "cc_chain_depth_max",         # max chain depth observed
    "cc_total_delay_mean",        # avg total downstream delay (min) across all legs
    "cc_total_delay_p75",         # p75 total downstream delay
    "cc_cascade_rate",            # fraction of sequences with ≥1 downstream late-aircraft leg
    "cc_recovery_rate",           # fraction where cascade fully recovers
    "cc_amplification_mean",      # total_downstream_delay / B_arr_delay ratio
    "cc_affected_airports_mean",  # avg unique airports hit downstream
    "cc_max_single_leg_delay",    # avg worst single downstream leg delay
]


def focal_loss(gamma: float = 2.0, alpha: float = 0.25):
    """
    LightGBM custom focal loss for rare-event handling.
    Reduces weight of easy negatives → focuses on hard/rare positives.
    gamma: focusing parameter (higher = more focus on hard examples)
    alpha: class balance weight for positives
    """
    def _focal_loss(y_true, y_pred):
        p   = 1.0 / (1.0 + np.exp(-y_pred))
        # Gradient and hessian for focal loss
        g   = np.where(y_true == 1,
                       -alpha * (1 - p) ** gamma * (gamma * p * np.log(p + 1e-9) + p - 1),
                        (1 - alpha) * p ** gamma * (gamma * (1 - p) * np.log(1 - p + 1e-9) - (1 - p) + 1))
        h   = np.where(y_true == 1,
                       alpha * (1 - p) ** gamma * (gamma ** 2 * p * (np.log(p + 1e-9) + 1) + gamma * (2 * p - 1) - p + 1),
                       (1 - alpha) * p ** gamma * (gamma ** 2 * (1 - p) * (np.log(1 - p + 1e-9) + 1) - gamma * (2 * p - 1) - (1 - p) + 1))
        return g, h.clip(1e-6, None)

    def _eval_auc(y_true, y_pred):
        return "auc", roc_auc_score(y_true, 1 / (1 + np.exp(-y_pred))), True

    return _focal_loss, _eval_auc


def load_data():
    path = os.path.join(PROC_DIR, "sequence_features_duty.parquet")
    if not os.path.exists(path):
        print("Duty features not found — run feature_engineering_duty.py first")
        sys.exit(1)
    df = pd.read_parquet(path)

    # Fix label: absolute threshold instead of pre-baked median-based 50/50 split
    df["target"] = (df["observed_bad_rate"] > RISK_THRESHOLD).astype(int)
    print(f"Target recomputed: {df['target'].mean():.1%} positive "
          f"(observed_bad_rate > {RISK_THRESHOLD})")

    # Join GSOM cols from sequence_features (duty parquet predates GSOM enrichment)
    GSOM_COLS = [
        "A_avg_wind_speed", "A_precip_days", "A_extreme_precip",
        "A_total_precip", "A_max_wind_gust",
        "B_avg_wind_speed", "B_precip_days", "B_extreme_precip",
        "B_total_precip", "B_max_wind_gust",
        "pair_max_avg_wind_speed", "pair_max_precip_days",
        "pair_max_extreme_precip", "pair_max_total_precip", "pair_max_max_wind_gust",
    ]
    missing_gsom = [c for c in GSOM_COLS if c not in df.columns]
    if missing_gsom:
        sf_base = pd.read_parquet(os.path.join(PROC_DIR, "sequence_features.parquet"))
        join_cols = ["airport_A", "airport_B", "Month", "Year"]
        avail = [c for c in missing_gsom if c in sf_base.columns]
        if avail:
            df = df.merge(sf_base[join_cols + avail], on=join_cols, how="left")
            print(f"GSOM cols backfilled from sequence_features: {avail[:3]}... ({len(avail)} total)")

    # Join DFW hub weather (reuse cached file from model.py if available)
    dfw_cache = os.path.join(PROC_DIR, "dfw_weather_monthly.parquet")
    if os.path.exists(dfw_cache):
        dfw = pd.read_parquet(dfw_cache)
        df = df.merge(dfw, on="Month", how="left")
        print(f"DFW weather joined ({len(dfw)} months)")

    # Join tail-chain features (aircraft rotation → crew duty proxy)
    tc_path = os.path.join(PROC_DIR, "tail_chain_features.parquet")
    if os.path.exists(tc_path):
        tc = pd.read_parquet(tc_path)
        df = df.merge(tc, on=["airport_A", "airport_B", "Month", "Year"], how="left")
        tc_coverage = df["tc_legs_before_mean"].notna().mean()
        print(f"Tail-chain features joined: {tc_coverage:.1%} coverage")
    else:
        print("Warning: tail_chain_features.parquet not found — run tail_chain_features.py")

    # Join multi-hop DFW cascade features (A→DFW→B→DFW→C→DFW→D→...)
    mhc_path = os.path.join(PROC_DIR, "multihop_cascade_features.parquet")
    if os.path.exists(mhc_path):
        mhc = pd.read_parquet(mhc_path)
        df = df.merge(mhc, on=["airport_A", "airport_B", "Month", "Year"], how="left")
        mhc_coverage = df["mhc_n_hops_mean"].notna().mean()
        print(f"Multi-hop DFW cascade features joined: {mhc_coverage:.1%} coverage")
    else:
        print("Info: multihop_cascade_features.parquet not found — run multihop_dfw_cascade.py")

    # Join full national cascade chain features (A→DFW→B→C→D→..., any dest) if available
    cc_path = os.path.join(PROC_DIR, "cascade_chain_features.parquet")
    if os.path.exists(cc_path):
        cc = pd.read_parquet(cc_path)
        df = df.merge(cc, on=["airport_A", "airport_B", "Month", "Year"], how="left")
        cc_coverage = df["cc_chain_depth_mean"].notna().mean()
        print(f"Full cascade chain features joined: {cc_coverage:.1%} coverage")
    else:
        print("Info: cascade_chain_features.parquet not found — run download_bts_full.py + cascade_chain_features.py when BTS is back up")

    # Join airport-level cascade propagation features
    ap_cascade_path = os.path.join(PROC_DIR, "airport_cascade_features.parquet")
    if os.path.exists(ap_cascade_path):
        ap = pd.read_parquet(ap_cascade_path)
        ap_cols = [c for c in ap.columns if c not in ("airport", "Month")]
        ap_a = ap.rename(columns={"airport": "airport_A",
                                   **{c: f"A_{c}" for c in ap_cols}})
        ap_b = ap.rename(columns={"airport": "airport_B",
                                   **{c: f"B_{c}" for c in ap_cols}})
        df = df.merge(ap_a, on=["airport_A", "Month"], how="left")
        df = df.merge(ap_b, on=["airport_B", "Month"], how="left")
        # Pair-level cascade interaction features
        df["pair_cascade_product"]  = df["A_ap_cascade_rate"] * df["B_ap_cascade_rate"]
        df["pair_max_cascade_rate"] = df[["A_ap_cascade_rate", "B_ap_cascade_rate"]].max(axis=1)
        print(f"Airport cascade features joined: "
              f"{df['A_ap_cascade_rate'].notna().mean():.1%} coverage")
    else:
        print("Warning: airport_cascade_features.parquet not found — run tail_chain_features.py")

    season_cols = [c for c in df.columns if c.startswith("season_")]
    all_features = BASE_FEATURES + DUTY_FEATURES + TAIL_CHAIN_FEATURES + season_cols
    feat_cols = [c for c in all_features if c in df.columns]
    missing = [c for c in all_features if c not in df.columns]
    if missing:
        print(f"Warning: {len(missing)} features missing: {missing[:5]}...")
    return df, feat_cols


def train_with_cv(df: pd.DataFrame, feat_cols: list[str]) -> lgb.LGBMClassifier:
    """
    TimeSeriesSplit CV on years (chronological), then final fit on all but 2024.
    """
    years = sorted(df["Year"].unique())
    print(f"Years: {years}")

    # TimeSeriesSplit by year index
    year_idx = df["Year"].map({y: i for i, y in enumerate(years)})
    tscv = TimeSeriesSplit(n_splits=len(years) - 1)

    cv_aucs, cv_aps = [], []
    print("\nTimeSeriesSplit cross-validation:")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(year_idx.values.reshape(-1, 1))):
        train_years = df.iloc[train_idx]["Year"].unique()
        val_years   = df.iloc[val_idx]["Year"].unique()
        X_tr = df.iloc[train_idx][feat_cols].astype(float)
        y_tr = df.iloc[train_idx]["target"].astype(int)
        X_va = df.iloc[val_idx][feat_cols].astype(float)
        y_va = df.iloc[val_idx]["target"].astype(int)

        m = lgb.LGBMClassifier(
            n_estimators=500, learning_rate=0.05, max_depth=6,
            num_leaves=63, subsample=0.8, colsample_bytree=0.8,
            class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
            device="cuda",
        )
        m.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
              callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(-1)])
        p = m.predict_proba(X_va)[:, 1]
        auc = roc_auc_score(y_va, p)
        ap  = average_precision_score(y_va, p)
        cv_aucs.append(auc)
        cv_aps.append(ap)
        print(f"  Fold {fold+1}: train={sorted(train_years)}  val={sorted(val_years)}  "
              f"AUC={auc:.4f}  AP={ap:.4f}")

    print(f"\nCV mean AUC: {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}")
    print(f"CV mean AP:  {np.mean(cv_aps):.4f} ± {np.std(cv_aps):.4f}")

    # Final model: train on all years except 2024
    print("\nFitting final model (train < 2024, holdout = 2024)...")
    train_mask = df["Year"] < 2024
    val_mask   = df["Year"] == 2024
    X_train = df[train_mask][feat_cols].astype(float)
    y_train = df[train_mask]["target"].astype(int)
    X_val   = df[val_mask][feat_cols].astype(float)
    y_val   = df[val_mask]["target"].astype(int)

    final = lgb.LGBMClassifier(
        n_estimators=1000, learning_rate=0.03, max_depth=6,
        num_leaves=63, subsample=0.8, colsample_bytree=0.8,
        class_weight="balanced", random_state=42, n_jobs=-1, verbose=-1,
        device="cuda",
    )
    final.fit(X_train, y_train, eval_set=[(X_val, y_val)],
              callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)])

    proba = final.predict_proba(X_val)[:, 1]
    roc = roc_auc_score(y_val, proba)
    ap  = average_precision_score(y_val, proba)
    print(f"\n--- 2024 Holdout ---")
    print(f"ROC-AUC:           {roc:.4f}")
    print(f"Average Precision: {ap:.4f}")
    print(classification_report(y_val, final.predict(X_val), target_names=["low_risk","high_risk"]))

    return final, feat_cols, cv_aucs, cv_aps


def compare_with_base(df, feat_cols, lgbm_model):
    """Compare LightGBM duty-aware vs base XGBoost on 2024 holdout."""
    import xgboost as xgb

    val = df[df["Year"] == 2024].copy()
    X_val = val[feat_cols].astype(float)
    y_val = val["target"].astype(int)

    # LightGBM duty-aware
    lgbm_proba = lgbm_model.predict_proba(X_val)[:, 1]
    lgbm_auc   = roc_auc_score(y_val, lgbm_proba)
    lgbm_ap    = average_precision_score(y_val, lgbm_proba)

    # Base XGBoost (base features only)
    base_model = xgb.XGBClassifier()
    base_model.load_model(os.path.join(PROC_DIR, "xgb_model.json"))
    base_feat_cols = base_model.get_booster().feature_names
    X_base = val[[c for c in base_feat_cols if c in val.columns]].astype(float)
    for c in base_feat_cols:
        if c not in X_base.columns:
            X_base[c] = 0.0
    X_base = X_base[base_feat_cols]
    base_proba = base_model.predict_proba(X_base)[:, 1]
    base_auc   = roc_auc_score(y_val, base_proba)
    base_ap    = average_precision_score(y_val, base_proba)

    print("\n" + "="*55)
    print("MODEL COMPARISON — 2024 Holdout")
    print("="*55)
    print(f"{'Model':<30} {'ROC-AUC':>8} {'Avg Prec':>10}")
    print(f"{'-'*50}")
    print(f"{'XGBoost (base, weather)':<30} {base_auc:>8.4f} {base_ap:>10.4f}")
    print(f"{'LightGBM (duty-aware)':<30} {lgbm_auc:>8.4f} {lgbm_ap:>10.4f}")
    delta_auc = lgbm_auc - base_auc
    delta_ap  = lgbm_ap  - base_ap
    print(f"{'Delta':<30} {delta_auc:>+8.4f} {delta_ap:>+10.4f}")

    # Decile calibration comparison
    val["lgbm_proba"] = lgbm_proba
    val["base_proba"] = base_proba
    val["lgbm_decile"] = pd.qcut(val["lgbm_proba"], q=10, labels=False, duplicates="drop")
    val["base_decile"] = pd.qcut(val["base_proba"], q=10, labels=False, duplicates="drop")

    lgbm_dec = val.groupby("lgbm_decile")["target"].mean()
    base_dec  = val.groupby("base_decile")["target"].mean()

    # Plot comparison
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Calibration
    ax = axes[0]
    ax.plot(range(10), base_dec.values,  "o-", color="steelblue", label=f"XGBoost base  (AUC={base_auc:.3f})")
    ax.plot(range(10), lgbm_dec.values, "s-", color="darkorange", label=f"LightGBM duty (AUC={lgbm_auc:.3f})")
    ax.set_xlabel("Risk decile (0=lowest, 9=highest)")
    ax.set_ylabel("Actual bad rate")
    ax.set_title("Decile calibration comparison\n2024 holdout")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # Feature importance — duty model
    ax = axes[1]
    imp = pd.DataFrame({
        "feature":    feat_cols,
        "importance": lgbm_model.feature_importances_,
    }).sort_values("importance", ascending=True).tail(20)
    colors = ["darkorange" if any(d in f for d in ["block","duty","cascade","aircraft","connection","dep_rate","dep_hour"])
              else "steelblue" for f in imp["feature"]]
    ax.barh(imp["feature"], imp["importance"], color=colors)
    ax.set_title("LightGBM feature importance\n(orange = new duty features)")
    ax.set_xlabel("Importance")
    ax.grid(axis="x", alpha=0.3)

    # Score distribution
    ax = axes[2]
    ax.hist(lgbm_proba[y_val == 0], bins=40, alpha=0.6, color="steelblue", label="Low risk (actual)")
    ax.hist(lgbm_proba[y_val == 1], bins=40, alpha=0.6, color="red",       label="High risk (actual)")
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Count")
    ax.set_title("LightGBM score distribution\n2024 holdout")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS, "lgbm_duty_comparison.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"\nComparison plot → {out}")

    return {"lgbm_auc": lgbm_auc, "lgbm_ap": lgbm_ap,
            "base_auc": base_auc, "base_ap": base_ap}


def main():
    print("Loading duty-enriched features...")
    df, feat_cols = load_data()
    print(f"Dataset: {df.shape}  |  Features: {len(feat_cols)}")
    print(f"Duty features present: {[f for f in DUTY_FEATURES if f in feat_cols]}")

    model, feat_cols, cv_aucs, cv_aps = train_with_cv(df, feat_cols)

    # Save model
    model_path = os.path.join(PROC_DIR, "lgbm_duty_model.txt")
    model.booster_.save_model(model_path)
    print(f"\nModel saved → {model_path}")

    compare_with_base(df, feat_cols, model)

    print("\nDone. New artifacts:")
    print("  data/processed/lgbm_duty_model.txt")
    print("  data/processed/plots/lgbm_duty_comparison.png")


if __name__ == "__main__":
    main()
