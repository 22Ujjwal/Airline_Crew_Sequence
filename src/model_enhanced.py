"""
Enhanced XGBoost model using enriched features:
  - Duty/fatigue features (from feature_engineering_duty.py)
  - Actual METAR weather for airports A, B, and DFW hub (from enrich_features.py)
  - Time-of-day signals (afternoon TS window, morning arrivals)

Run after:
  python src/feature_engineering.py
  python src/feature_engineering_duty.py
  python src/enrich_features.py
"""

import os
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")

# Base features (same as model.py)
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
]

# Duty features (from feature_engineering_duty.py)
DUTY_FEATURES = [
    "A_dep_hour_median", "A_late_dep_rate", "A_early_dep_rate",
    "A_avg_block_min", "A_late_aircraft_delay_rate", "A_avg_late_aircraft_min",
    "B_dep_hour_median", "B_late_dep_rate", "B_early_dep_rate",
    "B_avg_block_min", "B_late_aircraft_delay_rate", "B_avg_late_aircraft_min",
    "tight_connection_rate", "very_tight_rate",
    "cascade_risk", "total_duty_block_min", "duty_overrun_risk", "late_dep_sequence",
]

# Actual METAR weather features (from enrich_features.py)
METAR_FEATURES = [
    # DFW hub weather (all sequences pass through)
    "dfw_ts_rate", "dfw_fog_rate", "dfw_snow_rate",
    "dfw_low_ceil_rate", "dfw_avg_severity", "dfw_avg_wind_kt",
    # Airport A observed weather
    "A_wx_ts_rate", "A_wx_fog_rate", "A_wx_snow_rate",
    "A_wx_avg_severity", "A_wx_avg_wind_kt",
    # Airport B observed weather
    "B_wx_ts_rate", "B_wx_fog_rate", "B_wx_snow_rate",
    "B_wx_avg_severity", "B_wx_avg_wind_kt",
    # Pair-level METAR derived
    "pair_wx_ts_rate", "pair_wx_severity",
    "dfw_x_A_ts", "dfw_x_B_ts",
]

# Time-of-day features (from enrich_features.py)
TOD_FEATURES = [
    "B_afternoon_dep_rate",  # DFW→B departures in TS peak window (14-19h)
    "A_morning_arr_rate",    # A→DFW arrivals in morning (06-12h)
]


def load_features():
    path = os.path.join(PROCESSED_DIR, "sequence_features_enhanced.parquet")
    if not os.path.exists(path):
        raise FileNotFoundError(f"Run enrich_features.py first.\nExpected: {path}")
    df = pd.read_parquet(path)
    season_cols = [c for c in df.columns if c.startswith("season_")]
    print(f"Loaded enhanced features: {df.shape}")
    return df, season_cols


def build_feature_cols(df: pd.DataFrame, season_cols: list[str]) -> list[str]:
    all_candidates = BASE_FEATURES + DUTY_FEATURES + METAR_FEATURES + TOD_FEATURES + season_cols
    present = [c for c in all_candidates if c in df.columns]
    missing = [c for c in all_candidates if c not in df.columns]
    if missing:
        print(f"  Missing features (skipped): {missing}")
    print(f"  Using {len(present)} features total")
    return present


def train(df: pd.DataFrame, feature_cols: list[str]) -> xgb.XGBClassifier:
    # Fill NaN in new features with column median (airports without METAR data)
    df = df.copy()
    for col in feature_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    X = df[feature_cols].astype(float)
    y = df["target"].astype(int)

    neg, pos = (y == 0).sum(), (y == 1).sum()
    scale_pos_weight = neg / pos
    print(f"Class balance — neg: {neg:,}, pos: {pos:,}, spw: {scale_pos_weight:.2f}")

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

    train_mask = df["Year"] < df["Year"].max()
    val_mask   = df["Year"] == df["Year"].max()
    X_train, y_train = X[train_mask], y[train_mask]
    X_val,   y_val   = X[val_mask],   y[val_mask]
    print(f"Train: {len(X_train):,} rows | Val: {len(X_val):,} rows")

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=50)

    y_prob = model.predict_proba(X_val)[:, 1]
    y_pred = model.predict(X_val)

    print(f"\n--- Enhanced Model Validation ---")
    print(f"ROC-AUC:           {roc_auc_score(y_val, y_prob):.4f}")
    print(f"Average Precision: {average_precision_score(y_val, y_prob):.4f}")
    print(classification_report(y_val, y_pred, target_names=["low_risk", "high_risk"]))

    return model, df   # return df with filled NaN for scoring


def plot_feature_importance(model: xgb.XGBClassifier, feature_cols: list[str]):
    feat_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=True).tail(30)

    fig, ax = plt.subplots(figsize=(10, 10))
    colors = []
    for f in feat_df["feature"]:
        if f in METAR_FEATURES or f.startswith("dfw_"):
            colors.append("coral")
        elif f in DUTY_FEATURES:
            colors.append("mediumseagreen")
        elif f in TOD_FEATURES:
            colors.append("mediumpurple")
        else:
            colors.append("steelblue")
    ax.barh(feat_df["feature"], feat_df["importance"], color=colors)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor="steelblue",      label="Base BTS features"),
        Patch(facecolor="mediumseagreen", label="Duty/fatigue features"),
        Patch(facecolor="coral",          label="Actual METAR weather"),
        Patch(facecolor="mediumpurple",   label="Time-of-day features"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)
    ax.set_xlabel("Feature Importance (gain)")
    ax.set_title("Top 30 Feature Importances — Enhanced Model\n"
                 "Base + Duty + METAR + Time-of-Day")
    plt.tight_layout()
    out = os.path.join(PROCESSED_DIR, "feature_importance_enhanced.png")
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Feature importance plot saved → {out}")


def main():
    df, season_cols = load_features()
    feature_cols = build_feature_cols(df, season_cols)

    model, df_filled = train(df, feature_cols)

    model_path = os.path.join(PROCESSED_DIR, "xgb_model_enhanced.json")
    model.save_model(model_path)
    print(f"Model saved → {model_path}")

    plot_feature_importance(model, feature_cols)

    # Re-score all pairs with enhanced model
    X_all = df_filled[feature_cols].astype(float)
    df_filled = df_filled.copy()
    df_filled["risk_score"] = model.predict_proba(X_all)[:, 1]
    pair_scores = (
        df_filled.groupby(["airport_A", "airport_B", "Month"])
        .agg(
            avg_risk_score   = ("risk_score",        "mean"),
            max_risk_score   = ("risk_score",        "max"),
            n_sequences      = ("n_sequences",       "sum"),
            observed_bad_rate= ("observed_bad_rate", "mean"),
        )
        .reset_index()
        .sort_values("avg_risk_score", ascending=False)
    )
    out = os.path.join(PROCESSED_DIR, "pair_risk_scores_enhanced.parquet")
    pair_scores.to_parquet(out, index=False)
    print(f"\nEnhanced pair risk scores saved → {out}")
    print("\nTop 20 riskiest pairs (enhanced model):")
    print(pair_scores.head(20).to_string(index=False))


if __name__ == "__main__":
    main()
