"""
XGBoost model to classify high-risk crew sequences (A→DFW→B pairs).

Outputs:
  - Trained model saved to data/processed/xgb_model.json
  - Per-airport-pair risk scores saved to data/processed/pair_risk_scores.parquet
  - Feature importance plot
"""

import os
import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import StratifiedGroupKFold, cross_val_score
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelEncoder
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")

# Absolute observed_bad_rate threshold: pairs where >25% of sequences had weather
# disruption are labeled high-risk. More meaningful than median (which yields 50/50 split).
RISK_THRESHOLD = 0.25

FEATURE_COLS = [
    # Airport A BTS weather stats
    "A_weather_delay_rate", "A_weather_cancel_rate", "A_avg_weather_delay_min",
    "A_p75_weather_delay_min", "A_p95_weather_delay_min", "A_nas_delay_rate",
    "A_overall_weather_delay_rate", "A_overall_avg_weather_delay_min",
    # Airport B BTS weather stats
    "B_weather_delay_rate", "B_weather_cancel_rate", "B_avg_weather_delay_min",
    "B_p75_weather_delay_min", "B_p95_weather_delay_min", "B_nas_delay_rate",
    "B_overall_weather_delay_rate", "B_overall_avg_weather_delay_min",
    # Pair-level BTS features
    "pair_combined_weather_rate", "pair_max_weather_rate", "pair_min_weather_rate",
    "pair_weather_rate_sum", "pair_avg_weather_delay_min", "both_high_risk",
    # Temporal
    "Month", "is_spring_summer", "median_turnaround_min",
    # GSOM weather features (wind + precip; ~55% airports have station coverage)
    # XGBoost handles NaN natively — missing airports simply skip these splits
    "A_avg_wind_speed", "A_precip_days", "A_extreme_precip",
    "A_total_precip", "A_max_wind_gust",
    "B_avg_wind_speed", "B_precip_days", "B_extreme_precip",
    "B_total_precip", "B_max_wind_gust",
    "pair_max_avg_wind_speed", "pair_max_precip_days",
    "pair_max_extreme_precip", "pair_max_total_precip", "pair_max_max_wind_gust",
    # DFW hub weather (all sequences pass through DFW)
    "DFW_weather_delay_rate", "DFW_weather_cancel_rate",
    "DFW_avg_weather_delay_min", "DFW_p95_weather_delay_min",
    # Tail-chain: crew duty context from aircraft rotation (proxy for same crew)
    "tc_legs_before_mean", "tc_block_before_mean", "tc_duty_start_hour",
    "tc_total_duty_mean", "tc_total_duty_p75",
    "tc_fdp_util_mean", "tc_fdp_util_p75", "tc_fdp_overrun_rate",
    "tc_wocl_rate", "tc_legs_after_mean", "tc_legs_in_day_mean",
    "tc_downstream_rate", "tc_cascade_late_rate",
    "tc_cascade_late_min", "tc_cascade_amplif_mean",
    # Airport-level cascade propagation index
    "A_ap_cascade_rate", "A_ap_cascade_given_late",
    "B_ap_cascade_rate", "B_ap_cascade_given_late",
    "pair_cascade_product", "pair_max_cascade_rate",
    # Multi-hop DFW cascade: A→DFW→B→DFW→C→DFW→D
    "mhc_n_hops_mean", "mhc_n_hops_max",
    "mhc_total_late_min_mean", "mhc_total_late_min_p75",
    "mhc_cascade_hop_rate", "mhc_cascade_depth_mean",
    "mhc_unique_airports_mean", "mhc_recovery_rate",
    # Interaction features (computed in load_features)
    "A_B_weather_coincidence",   # joint airport risk (product)
    "B_given_A_risk_ratio",      # how much B compounds A's risk
    # Temporal lag: storm clustering month-over-month
    "prev_month_bad_rate",
]


def get_dfw_weather() -> pd.DataFrame:
    """
    Compute DFW monthly weather stats from raw BTS files.
    DFW is the hub for every sequence, so its weather is a shared risk factor
    not captured in airport_features (which only covers non-DFW airports).
    Results cached to data/processed/dfw_weather_monthly.parquet.
    """
    cache = os.path.join(PROCESSED_DIR, "dfw_weather_monthly.parquet")
    if os.path.exists(cache):
        return pd.read_parquet(cache)

    import glob
    files = sorted(glob.glob(os.path.join(RAW_DIR, "bts_all_dfw_*.parquet")))
    if not files:
        files = sorted(glob.glob(os.path.join(RAW_DIR, "bts_aa_dfw_*.parquet")))
    print(f"Computing DFW weather from {len(files)} raw files...")

    frames = []
    for f in files:
        df = pd.read_parquet(f)
        # All flights in dataset touch DFW (Origin or Dest)
        df["weather_delayed"] = (df["WeatherDelay"].fillna(0) >= 15).astype(int)
        df["weather_cancel"] = (
            (df["Cancelled"] == 1) & (df["CancellationCode"] == "B")
        ).astype(int)
        df["weather_delay_min"] = df["WeatherDelay"].fillna(0)
        frames.append(df[["Month", "weather_delayed", "weather_cancel", "weather_delay_min"]])

    combined = pd.concat(frames, ignore_index=True)
    dfw = (
        combined.groupby("Month")
        .agg(
            DFW_weather_delay_rate=("weather_delayed", "mean"),
            DFW_weather_cancel_rate=("weather_cancel", "mean"),
            DFW_avg_weather_delay_min=("weather_delay_min", "mean"),
            DFW_p95_weather_delay_min=("weather_delay_min", lambda x: x.quantile(0.95)),
        )
        .reset_index()
    )
    dfw.to_parquet(cache, index=False)
    print(f"DFW weather cached → {cache}")
    return dfw


def load_features():
    path = os.path.join(PROCESSED_DIR, "sequence_features.parquet")
    df = pd.read_parquet(path)

    # Fix label: use absolute threshold (>25% bad rate = high risk) instead of
    # the pre-baked median-based target which forces a meaningless 50/50 split.
    df["target"] = (df["observed_bad_rate"] > RISK_THRESHOLD).astype(int)
    print(f"Target recomputed: {df['target'].mean():.1%} positive "
          f"(observed_bad_rate > {RISK_THRESHOLD})")

    # Join DFW hub weather by month
    dfw = get_dfw_weather()
    df = df.merge(dfw, on="Month", how="left")

    # Join tail-chain features (crew duty proxy via aircraft rotation)
    tc_path = os.path.join(PROCESSED_DIR, "tail_chain_features.parquet")
    if os.path.exists(tc_path):
        tc = pd.read_parquet(tc_path)
        tc_cols = [c for c in tc.columns if c not in ("airport_A", "airport_B", "Month", "Year")]
        tc = tc[["airport_A", "airport_B", "Month", "Year"] + tc_cols]
        df = df.merge(tc, on=["airport_A", "airport_B", "Month", "Year"], how="left")
        print(f"Tail-chain joined: {len(tc_cols)} features, {tc['airport_A'].nunique()} airports covered")
    else:
        print("tail_chain_features.parquet not found — skipping tail-chain features")

    # Join airport-level cascade propagation (for both A and B)
    ap_path = os.path.join(PROCESSED_DIR, "airport_cascade_features.parquet")
    if os.path.exists(ap_path):
        ap = pd.read_parquet(ap_path)
        ap_feat_cols = [c for c in ap.columns if c not in ("airport", "Month")]
        for side in ("A", "B"):
            rename = {c: f"{side}_ap_{c}" for c in ap_feat_cols}
            merged = ap.rename(columns={"airport": f"airport_{side}", **rename})
            on_cols = [f"airport_{side}", "Month"]
            df = df.merge(merged[[f"airport_{side}", "Month"] + list(rename.values())],
                          on=on_cols, how="left")
        # Pair interaction features
        if "A_ap_cascade_rate" in df.columns and "B_ap_cascade_rate" in df.columns:
            df["pair_cascade_product"] = df["A_ap_cascade_rate"] * df["B_ap_cascade_rate"]
            df["pair_max_cascade_rate"] = df[["A_ap_cascade_rate", "B_ap_cascade_rate"]].max(axis=1)
        print(f"Airport cascade joined: {len(ap_feat_cols)} features per airport")
    else:
        print("airport_cascade_features.parquet not found — skipping airport cascade features")

    # Join multi-hop DFW cascade features
    mhc_path = os.path.join(PROCESSED_DIR, "multihop_cascade_features.parquet")
    if os.path.exists(mhc_path):
        mhc = pd.read_parquet(mhc_path)
        mhc_cols = [c for c in mhc.columns if c not in ("airport_A", "airport_B", "Month", "Year")]
        df = df.merge(mhc[["airport_A", "airport_B", "Month", "Year"] + mhc_cols],
                      on=["airport_A", "airport_B", "Month", "Year"], how="left")
        print(f"Multi-hop cascade joined: {len(mhc_cols)} features")
    else:
        print("multihop_cascade_features.parquet not found — skipping multi-hop cascade features")

    # ── Interaction features ───────────────────────────────────────────────
    # Joint weather risk: two airports simultaneously bad → higher compound risk
    if "A_weather_delay_rate" in df.columns and "B_weather_delay_rate" in df.columns:
        df["pair_max_weather_rate"] = df[["A_weather_delay_rate", "B_weather_delay_rate"]].max(axis=1)
        df["A_B_weather_coincidence"] = (
            df["A_weather_delay_rate"] * df["B_weather_delay_rate"]
        )
        df["B_given_A_risk_ratio"] = df["B_weather_delay_rate"] / (
            df["A_weather_delay_rate"] + 1e-6
        )

    # ── Temporal lag feature ───────────────────────────────────────────────
    # Storm clustering: last month's disruption rate predicts this month
    if "observed_bad_rate" in df.columns:
        df = df.sort_values(["airport_A", "airport_B", "Year", "Month"])
        df["prev_month_bad_rate"] = (
            df.groupby(["airport_A", "airport_B"])["observed_bad_rate"].shift(1)
        )

    season_cols = [c for c in df.columns if c.startswith("season_")]
    return df, season_cols


def train(df: pd.DataFrame, feature_cols: list[str]) -> xgb.XGBClassifier:
    X = df[feature_cols].astype(float)
    y = df["target"].astype(int)

    # Time-based splits:
    #   train   → years before second-to-last year
    #   val     → second-to-last year (early stopping signal)
    #   cal     → last 2 years (held out for isotonic calibration fitting)
    years      = sorted(df["Year"].unique())
    cal_cutoff = years[-2]          # last 2 years reserved for calibration
    val_year   = years[-3]          # year just before calibration window

    train_mask = df["Year"] < val_year
    val_mask   = df["Year"] == val_year
    cal_mask   = df["Year"] >= cal_cutoff

    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]

    print(f"Train:        {len(X_train):,} rows (Year < {val_year})")
    print(f"Val:          {len(X_val):,}   rows (Year == {val_year})")
    print(f"Cal holdout:  {cal_mask.sum():,}   rows (Year >= {cal_cutoff}, reserved for calibration)")

    # Class imbalance: weight minority class on training set only
    neg, pos = (y_train == 0).sum(), (y_train == 1).sum()
    scale_pos_weight = neg / pos
    print(f"Class balance (train) — neg: {neg:,}, pos: {pos:,}, spw: {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=42,
        n_jobs=-1,
        device="cuda",
        tree_method="hist",
    )

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    # ── Validation metrics ────────────────────────────────────────────────────
    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    auc = roc_auc_score(y_val, y_prob)
    ap  = average_precision_score(y_val, y_prob)

    # Top-k precision: how reliably does the model find the worst sequences?
    k            = max(1, int(0.10 * len(y_val)))
    top_k_idx    = np.argsort(y_prob)[-k:]
    top_k_prec   = float(y_val.iloc[top_k_idx].mean())

    # Expected Calibration Error (ECE): are predicted probabilities trustworthy?
    from sklearn.calibration import calibration_curve
    n_bins = 10
    frac_pos, mean_pred = calibration_curve(y_val, y_prob, n_bins=n_bins, strategy="quantile")
    ece = float(np.mean(np.abs(frac_pos - mean_pred)))

    print("\n--- Validation Results ---")
    print(f"ROC-AUC:            {auc:.4f}")
    print(f"Average Precision:  {ap:.4f}")
    print(f"Top-10% Precision:  {top_k_prec:.4f}  (model's best {k} calls, {top_k_prec:.1%} correct)")
    print(f"ECE:                {ece:.4f}  (0 = perfect calibration)")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=["low_risk", "high_risk"]))

    # Attach calibration holdout mask to model for use in fit_and_save_calibration
    model._cal_mask = cal_mask

    return model


def plot_feature_importance(model: xgb.XGBClassifier, feature_cols: list[str]):
    importances = model.feature_importances_
    feat_df = pd.DataFrame({"feature": feature_cols, "importance": importances})
    feat_df = feat_df.sort_values("importance", ascending=True).tail(20)

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.barh(feat_df["feature"], feat_df["importance"], color="steelblue")
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("Top 20 XGBoost Feature Importances\nCrew Sequence Weather Risk")
    plt.tight_layout()
    out = os.path.join(PROCESSED_DIR, "feature_importance.png")
    plt.savefig(out, dpi=150)
    print(f"Feature importance plot saved → {out}")


def score_all_pairs(model: xgb.XGBClassifier, df: pd.DataFrame, feature_cols: list[str]):
    """
    Aggregate sequence-level predictions into per-pair (airport_A, airport_B, Month)
    risk scores. This is the final deliverable: which pairs are risky, in which months.
    """
    X = df[feature_cols].astype(float)
    df = df.copy()
    df["risk_score"] = model.predict_proba(X)[:, 1]

    pair_scores = (
        df.groupby(["airport_A", "airport_B", "Month"])
        .agg(
            avg_risk_score=("risk_score", "mean"),
            max_risk_score=("risk_score", "max"),
            n_sequences=("n_sequences", "sum") if "n_sequences" in df.columns else ("risk_score", "count"),
            observed_bad_rate=("observed_bad_rate", "mean") if "observed_bad_rate" in df.columns else ("target", "mean"),
        )
        .reset_index()
        .sort_values("avg_risk_score", ascending=False)
    )

    out = os.path.join(PROCESSED_DIR, "pair_risk_scores.parquet")
    pair_scores.to_parquet(out, index=False)
    print(f"\nPair risk scores saved → {out}")
    print("\nTop 20 riskiest airport pairs:")
    print(pair_scores.head(20).to_string(index=False))
    return pair_scores


def fit_and_save_calibration(model: xgb.XGBClassifier, df: pd.DataFrame,
                              feature_cols: list[str]):
    """
    Fit isotonic calibration on the held-out calibration set (last 2 years).
    Using a holdout set prevents the calibration from memorising the training
    distribution, giving more honest probabilities on new route-months.

    Saves:
      calibration_isotonic.json  — (x=raw_score, y=calibrated) breakpoints
      gsom_month_medians.json    — legacy fallback (kept for predictor.py compatibility)
      pair_risk_scores.parquet   — updated with calibrated avg_risk_score
    """
    from sklearn.isotonic import IsotonicRegression

    cal_mask = getattr(model, "_cal_mask", df["Year"] >= df["Year"].max() - 1)

    X_cal = df.loc[cal_mask, feature_cols].astype(float)
    y_cal = df.loc[cal_mask, "target"].astype(int)
    obs_cal = df.loc[cal_mask, "observed_bad_rate"].astype(float)
    n_seq = df.loc[cal_mask, "n_sequences"].astype(float) if "n_sequences" in df.columns else None

    # Raw scores in calibration batches of 50k to avoid OOM
    raw_scores = np.zeros(len(X_cal), dtype=float)
    for start in range(0, len(X_cal), 50_000):
        batch = X_cal.iloc[start:start + 50_000]
        raw_scores[start:start + 50_000] = model.predict_proba(batch)[:, 1]

    # Fit isotonic on holdout: map raw XGBoost probability → observed bad rate
    iso = IsotonicRegression(out_of_bounds="clip")
    sample_weight = np.sqrt(n_seq.values) if n_seq is not None else None
    iso.fit(raw_scores, obs_cal.values, sample_weight=sample_weight)

    cal_out = {"x": iso.X_thresholds_.tolist(), "y": iso.y_thresholds_.tolist()}
    with open(os.path.join(PROCESSED_DIR, "calibration_isotonic.json"), "w") as f:
        json.dump(cal_out, f)
    print(f"Calibration saved ({len(iso.X_thresholds_)} breakpoints, "
          f"fitted on {cal_mask.sum():,} holdout rows)")

    # Calibrate pair_risk_scores
    def calibrate(p):
        return float(np.interp(p, iso.X_thresholds_, iso.y_thresholds_))

    prs_path = os.path.join(PROCESSED_DIR, "pair_risk_scores.parquet")
    if os.path.exists(prs_path):
        prs = pd.read_parquet(prs_path)
        prs["avg_risk_score_raw"] = prs["avg_risk_score"]
        prs["avg_risk_score"] = prs["avg_risk_score_raw"].apply(calibrate)
        prs.to_parquet(prs_path, index=False)
        print(f"pair_risk_scores.parquet updated with calibrated scores")

    # Save GSOM month medians as legacy fallback (predictor.py uses these if
    # openmeteo_airport_monthly.parquet is absent)
    gsom_cols = [c for c in feature_cols
                 if any(c.startswith(p) for p in ("A_avg_wind", "A_precip", "A_extreme",
                                                   "A_total_precip", "A_max_wind_gust"))]
    gsom_medians: dict = {}
    for col in gsom_cols:
        if col in df.columns:
            monthly = (
                df.groupby("Month")[col].median().dropna()
                .apply(lambda v: round(float(v), 4))
            )
            gsom_medians[col] = {str(k): v for k, v in monthly.items()}
    if gsom_medians:
        with open(os.path.join(PROCESSED_DIR, "gsom_month_medians.json"), "w") as f:
            json.dump(gsom_medians, f)


def main():
    df, season_cols = load_features()
    feature_cols = FEATURE_COLS + season_cols
    feature_cols = [c for c in feature_cols if c in df.columns]

    model = train(df, feature_cols)

    # Save model
    model_path = os.path.join(PROCESSED_DIR, "xgb_model.json")
    model.save_model(model_path)
    print(f"\nModel saved → {model_path}")

    plot_feature_importance(model, feature_cols)
    score_all_pairs(model, df, feature_cols)
    fit_and_save_calibration(model, df, feature_cols)


if __name__ == "__main__":
    main()
